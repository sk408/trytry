"""Disk-backed image cache for team logos and player headshots.

Images are downloaded once and stored in ``data/cache/``.  Subsequent
requests are served from disk instantly.

**Important**: The ``*_pixmap`` helpers used by the UI thread **never**
download.  They return a transparent placeholder when the image is not
yet cached.  Actual downloads happen only via the ``preload_*`` bulk
functions which run on background worker threads.

Public API
----------
- ``get_team_logo_pixmap(team_id, size)``  → QPixmap  (cache-only, instant)
- ``get_player_photo_pixmap(player_id, size)`` → QPixmap  (cache-only, instant)
- ``preload_team_logos(cb)``     – download all 30 logos (worker thread)
- ``preload_player_photos(ids, cb)`` – download rotation players (worker thread)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import requests

log = logging.getLogger(__name__)

# ── Cache directories ─────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TEAM_LOGO_DIR = _DATA_DIR / "cache" / "team_logos"
PLAYER_PHOTO_DIR = _DATA_DIR / "cache" / "player_photos"

TEAM_LOGO_DIR.mkdir(parents=True, exist_ok=True)
PLAYER_PHOTO_DIR.mkdir(parents=True, exist_ok=True)

# ── CDN URLs ──────────────────────────────────────────────────────────
# ESPN CDN for team logos (NBA CDN returns 403 Forbidden as of 2025)
_TEAM_LOGO_URL = "https://a.espncdn.com/i/teamlogos/nba/500/{espn_abbr}.png"
_TEAM_LOGO_URL_DARK = "https://a.espncdn.com/i/teamlogos/nba/500-dark/{espn_abbr}.png"

# NBA CDN still works for player headshots
_PLAYER_PHOTO_URL = (
    "https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"
)

# NBA API abbreviation → ESPN CDN abbreviation (lowercase)
# Most teams match (just lowercased), these are the exceptions:
_NBA_TO_ESPN_ABBR: dict[str, str] = {
    "GSW": "gs", "SAS": "sa", "NYK": "ny",
    "NOP": "no", "UTA": "utah", "WAS": "wsh",
}

_TIMEOUT = 10  # seconds per request
_DELAY = 0.4   # seconds between requests to avoid CDN throttling
_MAX_RETRIES = 2

# Shared session with keep-alive and browser-like headers
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Reuse a single Session for connection pooling + keep-alive."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "image/png,image/*,*/*;q=0.8",
        })
    return _session


# ── Low-level helpers ─────────────────────────────────────────────────

def _download(url: str, dest: Path, retries: int = _MAX_RETRIES) -> bool:
    """Download *url* to *dest* with retry.  Returns True on success."""
    sess = _get_session()
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, timeout=_TIMEOUT)
            if resp.status_code == 200 and len(resp.content) > 200:
                dest.write_bytes(resp.content)
                return True
            if resp.status_code == 429:
                # Rate-limited – back off and retry
                wait = 2.0 * attempt
                log.debug("image_cache: 429 rate-limited, waiting %.1fs", wait)
                time.sleep(wait)
                continue
            log.debug("image_cache: %s → status %s (attempt %d)",
                      url, resp.status_code, attempt)
        except requests.exceptions.Timeout:
            log.debug("image_cache: timeout %s (attempt %d)", url, attempt)
            time.sleep(1.0)
        except Exception as exc:
            log.debug("image_cache: error %s – %s (attempt %d)", url, exc, attempt)
            break  # non-retryable
    return False


def _cached_team_logo(team_id: int) -> Optional[Path]:
    """Return path if logo is already on disk, else None.  Never downloads."""
    dest = TEAM_LOGO_DIR / f"{team_id}.png"
    return dest if dest.exists() else None


def _cached_player_photo(player_id: int) -> Optional[Path]:
    """Return path if headshot is already on disk, else None.  Never downloads."""
    dest = PLAYER_PHOTO_DIR / f"{player_id}.png"
    return dest if dest.exists() else None


# ── Public: QPixmap helpers (CACHE-ONLY – never block) ────────────────

def get_team_logo_pixmap(team_id: int, size: int = 48):
    """Return a QPixmap for the team logo, scaled to *size* px.

    **Cache-only** – returns a transparent placeholder if the logo has
    not been preloaded yet.  Call ``preload_team_logos()`` from a worker
    thread first.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap

    path = _cached_team_logo(team_id)
    if path:
        pm = QPixmap(str(path))
        if not pm.isNull():
            return pm.scaled(
                size, size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
    # Transparent fallback
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    return pm


def get_player_photo_pixmap(player_id: int, size: int = 36):
    """Return a QPixmap for the player headshot, scaled to *size* px height.

    **Cache-only** – returns a transparent placeholder if not yet cached.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap

    path = _cached_player_photo(player_id)
    if path:
        pm = QPixmap(str(path))
        if not pm.isNull():
            return pm.scaledToHeight(
                size,
                Qt.TransformationMode.SmoothTransformation,
            )
    pm = QPixmap(1, 1)
    pm.fill(Qt.GlobalColor.transparent)
    return pm


# ── Bulk preload (run on WORKER THREADS only) ─────────────────────────

def preload_team_logos(progress_cb=None) -> int:
    """Download logos for all 30 teams.  Returns count of newly downloaded.

    Tries multiple CDN URL patterns and retries on failure.
    Adds a delay between requests to avoid rate-limiting.
    """
    from src.database.db import get_conn

    cb = progress_cb or (lambda _: None)

    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT team_id, abbreviation FROM teams ORDER BY abbreviation"
            ).fetchall()
    except Exception:
        return 0

    downloaded = 0
    skipped = 0
    failed = 0
    total = len(rows)

    for i, (tid, abbr) in enumerate(rows):
        dest = TEAM_LOGO_DIR / f"{tid}.png"
        if dest.exists():
            skipped += 1
            continue

        # Convert NBA abbreviation → ESPN CDN abbreviation (lowercase)
        espn_abbr = _NBA_TO_ESPN_ABBR.get(abbr, abbr).lower()

        # Try primary ESPN URL, then dark variant as fallback
        success = False
        for url in (
            _TEAM_LOGO_URL.format(espn_abbr=espn_abbr),
            _TEAM_LOGO_URL_DARK.format(espn_abbr=espn_abbr),
        ):
            if _download(url, dest):
                success = True
                break
            time.sleep(_DELAY)

        if success:
            downloaded += 1
            cb(f"  ✓ {abbr} logo downloaded")
        else:
            failed += 1
            cb(f"  ✗ {abbr} logo failed (team_id={tid})")

        # Progress update
        if (i + 1) % 5 == 0 or i + 1 == total:
            cb(f"Team logos: {i + 1}/{total} checked"
               f" ({downloaded} new, {skipped} cached, {failed} failed)")

        # Delay between requests to avoid throttling
        time.sleep(_DELAY)

    return downloaded


def preload_player_photos(
    player_ids: Optional[List[int]] = None,
    progress_cb=None,
) -> int:
    """Download headshots for active rotation players.

    By default fetches the top-minutes players per team (up to ~300),
    which covers everyone who appears in matchup/players views.
    """
    from src.database.db import get_conn

    cb = progress_cb or (lambda _: None)

    if player_ids is None:
        try:
            with get_conn() as conn:
                # Top 10 by minutes per team = ~300 players
                rows = conn.execute(
                    """
                    SELECT player_id FROM (
                        SELECT
                            p.player_id,
                            p.team_id,
                            COALESCE(AVG(ps.minutes), 0) AS avg_min,
                            ROW_NUMBER() OVER (
                                PARTITION BY p.team_id
                                ORDER BY COALESCE(AVG(ps.minutes), 0) DESC
                            ) AS rn
                        FROM players p
                        LEFT JOIN player_stats ps ON ps.player_id = p.player_id
                        WHERE p.team_id IS NOT NULL
                        GROUP BY p.player_id, p.team_id
                    )
                    WHERE rn <= 10 AND avg_min > 0
                    ORDER BY avg_min DESC
                    """
                ).fetchall()
            player_ids = [r[0] for r in rows]
        except Exception:
            return 0

    downloaded = 0
    skipped = 0
    failed = 0
    total = len(player_ids)
    cb(f"Player photos: {total} players to check")

    for i, pid in enumerate(player_ids):
        dest = PLAYER_PHOTO_DIR / f"{pid}.png"
        if dest.exists():
            skipped += 1
            continue

        url = _PLAYER_PHOTO_URL.format(player_id=pid)
        if _download(url, dest):
            downloaded += 1
        else:
            failed += 1

        # Progress every 20 players
        if (i + 1) % 20 == 0 or i + 1 == total:
            cb(f"Player photos: {i + 1}/{total} checked"
               f" ({downloaded} new, {skipped} cached, {failed} failed)")

        # Delay between requests
        time.sleep(_DELAY)

    return downloaded
