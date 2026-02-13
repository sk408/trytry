from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from queue import Queue
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "nba_analytics.db"
_SQLITE_BUSY_TIMEOUT_MS = 30_000
_DB_LOCK_TIMEOUT_S = 30.0
_db_use_lock = threading.RLock()
_db_state_lock = threading.Lock()
_active_db_users = 0
_db_task_queue: Queue["_QueuedDbTask"] = Queue()
_db_task_worker_lock = threading.Lock()
_db_task_worker: Optional[threading.Thread] = None
_db_pending_tasks: dict[str, Future] = {}


def _prepare_path() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DbLockState:
    active_users: int
    in_use: bool


@dataclass
class _QueuedDbTask:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future: Future
    dedupe_key: Optional[str] = None


class _ManagedConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._release_slot = None
        self._slot_released = False

    def _attach_release_slot(self, release_slot) -> None:
        self._release_slot = release_slot

    def _release_db_slot_once(self) -> None:
        if self._slot_released:
            return
        self._slot_released = True
        if self._release_slot is not None:
            self._release_slot()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._release_db_slot_once()


def _acquire_db_slot(timeout_s: float = _DB_LOCK_TIMEOUT_S) -> None:
    acquired = _db_use_lock.acquire(timeout=timeout_s)
    if not acquired:
        raise TimeoutError("Database access lock timeout: database is currently busy.")
    global _active_db_users
    with _db_state_lock:
        _active_db_users += 1


def _release_db_slot() -> None:
    global _active_db_users
    with _db_state_lock:
        _active_db_users = max(0, _active_db_users - 1)
    _db_use_lock.release()


def get_db_lock_state() -> DbLockState:
    with _db_state_lock:
        users = _active_db_users
    return DbLockState(active_users=users, in_use=users > 0)


def is_db_in_use() -> bool:
    return get_db_lock_state().in_use


def _run_db_task_worker() -> None:
    while True:
        task = _db_task_queue.get()
        try:
            if task.future.cancelled():
                continue
            result = task.fn(*task.args, **task.kwargs)
            if not task.future.done():
                task.future.set_result(result)
        except Exception as exc:
            if not task.future.done():
                task.future.set_exception(exc)
        finally:
            if task.dedupe_key:
                with _db_task_worker_lock:
                    _db_pending_tasks.pop(task.dedupe_key, None)


def _ensure_db_task_worker() -> None:
    global _db_task_worker
    with _db_task_worker_lock:
        if _db_task_worker is not None and _db_task_worker.is_alive():
            return
        _db_task_worker = threading.Thread(
            target=_run_db_task_worker,
            daemon=True,
            name="db-task-queue-worker",
        )
        _db_task_worker.start()


def queue_db_call(
    fn: Callable[..., Any],
    *args: Any,
    dedupe_key: Optional[str] = None,
    **kwargs: Any,
) -> Future:
    """Queue a callable that performs DB work and run it when available.

    Tasks are processed FIFO on a single background worker. If ``dedupe_key``
    is provided, only one pending task with that key is allowed.
    """
    _ensure_db_task_worker()
    with _db_task_worker_lock:
        if dedupe_key:
            existing = _db_pending_tasks.get(dedupe_key)
            if existing is not None and not existing.done():
                return existing
        future: Future = Future()
        _db_task_queue.put(
            _QueuedDbTask(
                fn=fn,
                args=args,
                kwargs=kwargs,
                future=future,
                dedupe_key=dedupe_key,
            )
        )
        if dedupe_key:
            _db_pending_tasks[dedupe_key] = future
        return future


def connect(timeout_s: float = _DB_LOCK_TIMEOUT_S) -> sqlite3.Connection:
    """Create a SQLite connection with foreign keys enabled."""
    _prepare_path()
    _acquire_db_slot(timeout_s=timeout_s)
    try:
        conn: _ManagedConnection = sqlite3.connect(
            DB_PATH,
            timeout=max(1.0, timeout_s),
            factory=_ManagedConnection,
        )
        conn._attach_release_slot(_release_db_slot)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS};")
        return conn
    except Exception:
        _release_db_slot()
        raise


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()
