"""Centralised notification service.

All monitors push notifications through this service.  It persists them
in the SQLite ``notifications`` table and (on the desktop) emits a Qt
signal so the system tray and bell widget can react immediately.
"""
from __future__ import annotations

import threading
import os
import json
import subprocess
import urllib.request
import urllib.error
from urllib.parse import urlparse
from typing import Callable, List, Optional

from src.database.db import get_conn
from src.notifications.models import Notification

# ── Module-level singleton ──

_lock = threading.Lock()

# Listeners: callables invoked on the *calling* thread when a new
# notification is added.  The desktop app registers a Qt-signal forwarder;
# the web app leaves this empty and polls the DB instead.
_listeners: List[Callable[[Notification], None]] = []


# ── Public API ──


def add(notification: Notification) -> Notification:
    """Persist a notification and notify listeners.  Returns it with ``id`` set."""
    with _lock:
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO notifications
                   (category, severity, title, body, data, created_at, read)
                   VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (
                    notification.category,
                    notification.severity,
                    notification.title,
                    notification.body,
                    notification.data_json(),
                    notification.created_at,
                ),
            )
            conn.commit()
            notification.id = cur.lastrowid
        listeners_snapshot = list(_listeners)
    # Fire listeners *outside* the lock so they can safely call back in.
    for fn in listeners_snapshot:
        try:
            fn(notification)
        except Exception:
            pass
    # Optional external push fan-out (non-blocking).
    threading.Thread(
        target=_dispatch_external_push,
        args=(notification,),
        daemon=True,
    ).start()
    # Best-effort local OS notification (desktop).
    threading.Thread(
        target=_dispatch_local_os_push,
        args=(notification,),
        daemon=True,
    ).start()
    return notification


def get_unread(limit: int = 50) -> List[Notification]:
    """Return up to *limit* unread notifications, newest first."""
    return _query("SELECT * FROM notifications WHERE read = 0 ORDER BY id DESC LIMIT ?", (limit,))


def get_recent(limit: int = 30) -> List[Notification]:
    """Return the most recent notifications regardless of read status."""
    return _query("SELECT * FROM notifications ORDER BY id DESC LIMIT ?", (limit,))


def unread_count() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) FROM notifications WHERE read = 0").fetchone()
        return row[0] if row else 0


def mark_read(notification_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE notifications SET read = 1 WHERE id = ?", (notification_id,))
        conn.commit()


def mark_all_read() -> None:
    with get_conn() as conn:
        conn.execute("UPDATE notifications SET read = 1 WHERE read = 0")
        conn.commit()


def register_listener(fn: Callable[[Notification], None]) -> None:
    """Register a callback invoked whenever a notification is added."""
    with _lock:
        if fn not in _listeners:
            _listeners.append(fn)


def unregister_listener(fn: Callable[[Notification], None]) -> None:
    with _lock:
        try:
            _listeners.remove(fn)
        except ValueError:
            pass


# ── Internal helpers ──


def _row_to_notification(row: tuple) -> Notification:
    # Column order: id, category, severity, title, body, data, created_at, read
    return Notification(
        id=row[0],
        category=row[1],
        severity=row[2],
        title=row[3],
        body=row[4],
        data=Notification.data_from_json(row[5]),
        created_at=row[6],
        read=bool(row[7]),
    )


def _query(sql: str, params: tuple = ()) -> List[Notification]:
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_notification(r) for r in rows]


def _dispatch_external_push(notification: Notification) -> None:
    """Push notifications to a webhook if configured.

    Set NOTIFY_WEBHOOK_URL (and optional NOTIFY_WEBHOOK_TOKEN) to enable.
    """
    _dispatch_external_push_impl(notification)


def _dispatch_external_push_impl(notification: Notification) -> bool:
    url = (os.getenv("NOTIFY_WEBHOOK_URL") or "").strip()
    if not url:
        # Convenience path for mobile pushes via ntfy:
        # set NOTIFY_NTFY_TOPIC=my-topic
        topic = (os.getenv("NOTIFY_NTFY_TOPIC") or "").strip().strip("/")
        if topic:
            ntfy_base = (os.getenv("NOTIFY_NTFY_BASE_URL") or "https://ntfy.sh").strip().rstrip("/")
            url = f"{ntfy_base}/{topic}"
    if not url:
        return False
    token = (os.getenv("NOTIFY_WEBHOOK_TOKEN") or "").strip()
    parsed = urlparse(url)
    is_ntfy = "ntfy.sh" in (parsed.netloc or "").lower()
    headers = {}
    if is_ntfy:
        body = notification.body.encode("utf-8")
        headers["Title"] = notification.title
        prio_map = {"critical": "urgent", "warning": "high", "info": "default"}
        headers["Priority"] = prio_map.get(notification.severity, "default")
        headers["Tags"] = notification.category
    else:
        payload = {
            "category": notification.category,
            "severity": notification.severity,
            "title": notification.title,
            "body": notification.body,
            "created_at": notification.created_at,
            "data": notification.data,
        }
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _dispatch_local_os_push(notification: Notification) -> None:
    """Best-effort local desktop notification.

    Toggle with NOTIFY_LOCAL_OS=0 (default is enabled).
    """
    _dispatch_local_os_push_impl(notification)


def _dispatch_local_os_push_impl(notification: Notification) -> bool:
    enabled = (os.getenv("NOTIFY_LOCAL_OS") or "1").strip().lower()
    if enabled in {"0", "false", "off", "no"}:
        return False

    title = (notification.title or "NBA Betting Analytics").strip()
    body = (notification.body or "").strip()
    if not body:
        body = notification.category

    # 1) Preferred cross-platform path if installed.
    try:
        from plyer import notification as plyer_notification

        plyer_notification.notify(
            title=title,
            message=body[:300],
            app_name="NBA Betting Analytics",
            timeout=8,
        )
        return True
    except Exception:
        pass

    # 2) Windows fallback via BurntToast if installed.
    if os.name == "nt":
        try:
            safe_title = title.replace('"', "'")
            safe_body = body[:350].replace('"', "'")
            ps = (
                "if (Get-Module -ListAvailable -Name BurntToast) { "
                "Import-Module BurntToast -ErrorAction SilentlyContinue; "
                f"New-BurntToastNotification -Text \"{safe_title}\", \"{safe_body}\" | Out-Null "
                "}"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return True
        except Exception:
            return False
    return False


def run_notification_diagnostics() -> dict:
    """Run end-to-end notification diagnostics and return path results."""
    test_notification = Notification(
        category=Notification.CATEGORY_INSIGHT,
        severity=Notification.SEVERITY_INFO,
        title="Notification Diagnostics Test",
        body="Testing in-app, local OS, and external push delivery paths.",
        data={"source": "admin_diagnostics"},
    )
    result: dict = {
        "db_write": False,
        "listeners_registered": 0,
        "local_os_enabled": (os.getenv("NOTIFY_LOCAL_OS") or "1").strip().lower() not in {"0", "false", "off", "no"},
        "external_configured": bool((os.getenv("NOTIFY_WEBHOOK_URL") or "").strip() or (os.getenv("NOTIFY_NTFY_TOPIC") or "").strip()),
        "local_os_push_sent": False,
        "external_push_sent": False,
        "error": "",
    }
    try:
        with _lock:
            result["listeners_registered"] = len(_listeners)
        add(test_notification)
        result["db_write"] = True
    except Exception as exc:
        result["error"] = f"DB/in-app notification failed: {exc}"
        return result

    try:
        result["local_os_push_sent"] = _dispatch_local_os_push_impl(test_notification)
    except Exception:
        result["local_os_push_sent"] = False

    try:
        result["external_push_sent"] = _dispatch_external_push_impl(test_notification)
    except Exception:
        result["external_push_sent"] = False

    return result
