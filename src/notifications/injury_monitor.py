"""Background injury monitor.

Polls injury sources periodically, diffs against the previous snapshot,
and fires notifications for high-impact player changes.  Also syncs the
DB so that matchup / gamecast views immediately reflect the latest data.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.notifications.models import Notification
from src.notifications import service as notif_service

log = logging.getLogger(__name__)

# MPG threshold that qualifies a player as "high-impact" (proxy for starter).
HIGH_IMPACT_MPG = 20.0

# ── Snapshot type: frozenset of (player_lower, team_lower, status_upper) ──
_SnapshotKey = Tuple[str, str]  # (player, team)


class InjuryMonitor:
    """Stateful injury diff engine.

    Call :meth:`check` periodically (e.g. every 5 min).  On the first call
    it establishes a baseline; subsequent calls diff against it and fire
    notifications for meaningful changes.
    """

    def __init__(self) -> None:
        self._baseline: Dict[Tuple[str, str], str] = {}  # (player, team) -> status
        self._initialised = False
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────
    def check(
        self,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> List[Notification]:
        """Fetch injuries, diff, sync DB, and return new notifications."""
        progress = progress_cb or (lambda _: None)
        notifications: List[Notification] = []

        try:
            # 1. Fetch current injuries
            from src.data.injury_scraper import get_all_injuries
            current_raw = get_all_injuries(timeout=15, progress_cb=progress)
            if not current_raw:
                progress("Injury monitor: no data from sources")
                return []

            # Build lookup: (player_lower, team_lower) -> entry dict
            current: Dict[Tuple[str, str], Dict] = {}
            for entry in current_raw:
                key = (
                    entry.get("player", "").strip().lower(),
                    entry.get("team", "").strip().lower(),
                )
                if key[0]:
                    current[key] = entry

            with self._lock:
                if not self._initialised:
                    # First run — just store baseline
                    self._baseline = {k: v.get("status", "").upper() for k, v in current.items()}
                    self._initialised = True
                    progress(f"Injury monitor: baseline set with {len(self._baseline)} entries")
                    # Still sync the DB on the first run
                    self._sync_db(progress)
                    return []

                # 2. Diff against baseline
                old_keys = set(self._baseline.keys())
                new_keys = set(current.keys())

                # New injuries (player was not on the report before)
                added = new_keys - old_keys
                # Status changed
                changed: Set[Tuple[str, str]] = set()
                for key in new_keys & old_keys:
                    new_status = current[key].get("status", "").upper()
                    if new_status != self._baseline.get(key, ""):
                        changed.add(key)

                # 3. Build notifications for high-impact players
                mpg_lookup = self._get_mpg_lookup()

                for key in added | changed:
                    entry = current[key]
                    player_name = entry.get("player", key[0])
                    team_name = entry.get("team", key[1])
                    status = entry.get("status", "Unknown")
                    injury_desc = entry.get("injury", "")
                    mpg = mpg_lookup.get(key[0], 0.0)

                    if mpg < HIGH_IMPACT_MPG:
                        continue  # skip low-minute players

                    severity = self._status_severity(status)
                    verb = "now listed as" if key in added else "status changed to"
                    body_parts = [f"{player_name} ({team_name}) {verb} {status}"]
                    if injury_desc:
                        body_parts.append(f"Injury: {injury_desc}")
                    body_parts.append(f"MPG: {mpg:.1f}")

                    n = Notification(
                        category=Notification.CATEGORY_INJURY,
                        severity=severity,
                        title=f"Injury Alert: {player_name}",
                        body=" | ".join(body_parts),
                        data={
                            "player": player_name,
                            "team": team_name,
                            "status": status,
                            "injury": injury_desc,
                            "mpg": round(mpg, 1),
                        },
                    )
                    notif_service.add(n)
                    notifications.append(n)
                    progress(f"  ALERT: {n.title} -- {status}")

                # Update baseline
                self._baseline = {k: v.get("status", "").upper() for k, v in current.items()}

            # 4. Sync DB so UI reflects latest
            self._sync_db(progress)

            if notifications:
                progress(f"Injury monitor: {len(notifications)} new alert(s)")
            else:
                progress("Injury monitor: no high-impact changes")

        except Exception as exc:
            log.exception("Injury monitor error")
            progress(f"Injury monitor error: {exc}")

        return notifications

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _sync_db(progress: Callable[[str], None]) -> None:
        """Re-run the injury sync so the DB flags are up to date."""
        try:
            from src.database.db import is_db_in_use, queue_db_call
            from src.data.sync_service import sync_injuries
            if is_db_in_use():
                queue_db_call(
                    sync_injuries,
                    progress_cb=progress,
                    dedupe_key="injury_monitor:sync_injuries",
                )
                progress("Injury monitor: DB is busy, queued injury sync")
                return
            sync_injuries(progress_cb=progress)
        except Exception as exc:
            log.warning("Injury monitor: DB sync failed: %s", exc)

    @staticmethod
    def _get_mpg_lookup() -> Dict[str, float]:
        """Return {player_name_lower: mpg} for all players with stats.

        Keys are normalised to ASCII (diacriticals stripped) so lookups
        from injury-scraper names (which are plain ASCII) succeed for
        players like Jokić / Dončić / Porziņģis.
        """
        try:
            from src.database.db import get_conn
            from src.data.sync_service import _strip_diacriticals
            with get_conn() as conn:
                rows = conn.execute(
                    """SELECT p.name, AVG(ps.minutes)
                       FROM players p
                       JOIN player_stats ps ON ps.player_id = p.player_id
                       GROUP BY p.player_id"""
                ).fetchall()
                return {
                    _strip_diacriticals(r[0]).lower(): r[1]
                    for r in rows if r[1]
                }
        except Exception:
            return {}

    @staticmethod
    def _status_severity(status: str) -> str:
        s = status.upper()
        if s == "OUT":
            return Notification.SEVERITY_CRITICAL
        if s in ("DOUBTFUL", "QUESTIONABLE"):
            return Notification.SEVERITY_WARNING
        return Notification.SEVERITY_INFO


# ── Convenience singleton for the desktop app ──

_instance: Optional[InjuryMonitor] = None
_instance_lock = threading.Lock()


def get_injury_monitor() -> InjuryMonitor:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = InjuryMonitor()
        return _instance
