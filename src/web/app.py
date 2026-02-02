from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.analytics.backtester import BacktestResults, run_backtest
from src.analytics.live_recommendations import build_live_recommendations
from src.analytics.prediction import predict_matchup
from src.analytics.stats_engine import get_scheduled_games
from src.data.sync_service import (
    full_sync,
    sync_injuries,
    sync_injury_history,
    sync_live_scores,
    sync_schedule,
)
from src.database import migrations
from src.database.db import DB_PATH, get_conn
from src.web.player_utils import get_position_display, load_injured_with_stats, load_players_df


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="NBA Betting Analytics (Web)")
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


@app.on_event("startup")
def startup() -> None:
    migrations.init_db()


def _df_to_records(df) -> List[dict]:
    if df is None:
        return []
    return [dict(row._asdict()) for row in df.itertuples(index=False)]


def _team_lookup() -> dict[int, str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT team_id, abbreviation FROM teams").fetchall()
    return {int(tid): abbr for tid, abbr in rows}


def _team_list() -> List[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT team_id, abbreviation, name FROM teams ORDER BY abbreviation").fetchall()
    return [
        {"id": int(tid), "abbr": abbr, "name": name}
        for tid, abbr, name in rows
    ]


def _team_players(team_id: int) -> List[int]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT player_id FROM players WHERE team_id = ? AND (is_injured = 0 OR is_injured IS NULL)",
            (team_id,),
        ).fetchall()
    return [int(pid) for (pid,) in rows]


def _run_with_progress(func, *, progress_label: str) -> tuple[str, List[str]]:
    messages: List[str] = []

    def _cb(msg: str) -> None:
        messages.append(msg)

    try:
        func(progress_cb=_cb)
        status = f"{progress_label} complete"
    except TypeError:
        func()
        status = f"{progress_label} complete"
    except Exception as exc:  # pragma: no cover - defensive
        status = f"{progress_label} failed: {exc}"
        messages.append(str(exc))
    return status, messages


@app.get("/", response_class=HTMLResponse)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, status: str | None = None, logs: List[str] | None = None) -> HTMLResponse:
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "status": status or "Ready", "logs": logs or []},
    )


@app.post("/dashboard/sync", response_class=HTMLResponse)
async def dashboard_sync(request: Request) -> HTMLResponse:
    status, logs = await asyncio.to_thread(_run_with_progress, full_sync, progress_label="Sync")
    return await dashboard(request, status=status, logs=logs)


@app.post("/dashboard/injuries", response_class=HTMLResponse)
async def dashboard_injuries(request: Request) -> HTMLResponse:
    status, logs = await asyncio.to_thread(_run_with_progress, sync_injuries, progress_label="Injury sync")
    return await dashboard(request, status=status, logs=logs)


@app.post("/dashboard/injury-history", response_class=HTMLResponse)
async def dashboard_injury_history(request: Request) -> HTMLResponse:
    status, logs = await asyncio.to_thread(_run_with_progress, sync_injury_history, progress_label="Injury history")
    return await dashboard(request, status=status, logs=logs)


# --- Server-Sent Events (SSE) for real-time progress ---

async def _stream_sync(sync_func, label: str) -> AsyncGenerator[str, None]:
    """Run a sync function and stream progress via SSE."""
    msg_queue: queue.Queue[str | None] = queue.Queue()

    def progress_cb(msg: str) -> None:
        msg_queue.put(msg)

    def run_sync() -> None:
        try:
            try:
                sync_func(progress_cb=progress_cb)
            except TypeError:
                sync_func()
            msg_queue.put(f"[DONE] {label} complete")
        except Exception as exc:
            msg_queue.put(f"[ERROR] {label} failed: {exc}")
        finally:
            msg_queue.put(None)  # Signal end

    thread = threading.Thread(target=run_sync, daemon=True)
    thread.start()

    while True:
        try:
            msg = await asyncio.to_thread(msg_queue.get, timeout=0.5)
            if msg is None:
                break
            yield f"data: {msg}\n\n"
        except Exception:
            if not thread.is_alive():
                break


@app.get("/api/sync/data")
async def stream_sync_data() -> StreamingResponse:
    """Stream data sync progress via SSE."""
    return StreamingResponse(
        _stream_sync(full_sync, "Data sync"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sync/injuries")
async def stream_sync_injuries() -> StreamingResponse:
    """Stream injury sync progress via SSE."""
    return StreamingResponse(
        _stream_sync(sync_injuries, "Injury sync"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sync/injury-history")
async def stream_sync_injury_history() -> StreamingResponse:
    """Stream injury history build progress via SSE."""
    return StreamingResponse(
        _stream_sync(sync_injury_history, "Injury history"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/backtest")
async def stream_backtest(
    home_team_id: int | None = None,
    away_team_id: int | None = None,
    use_injuries: str | None = None,
) -> StreamingResponse:
    """Stream backtest progress via SSE."""
    use_injuries_flag = True
    if use_injuries is not None:
        use_injuries_flag = str(use_injuries).lower() not in {"0", "false", "off"}

    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_bt() -> None:
            try:
                results = run_backtest(
                    5, home_team_id, away_team_id, progress_cb, use_injuries_flag
                )
                # Send results summary
                msg_queue.put(f"[RESULT] games={results.total_games}")
                msg_queue.put(f"[RESULT] spread_acc={results.overall_spread_accuracy:.1f}")
                msg_queue.put(f"[RESULT] total_acc={results.overall_total_accuracy:.1f}")
                msg_queue.put("[DONE] Backtest complete")
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_bt, daemon=True)
        thread.start()

        while True:
            try:
                msg = await asyncio.to_thread(msg_queue.get, timeout=0.5)
                if msg is None:
                    break
                yield f"data: {msg}\n\n"
            except Exception:
                if not thread.is_alive():
                    break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/live", response_class=HTMLResponse)
async def live(request: Request) -> HTMLResponse:
    error = None
    await asyncio.to_thread(sync_live_scores)
    try:
        recs = build_live_recommendations()
    except Exception as exc:  # pragma: no cover - network dependent
        recs = []
        error = str(exc)
    return templates.TemplateResponse(
        "live.html",
        {"request": request, "recs": recs, "error": error},
    )


@app.get("/players", response_class=HTMLResponse)
async def players(request: Request) -> HTMLResponse:
    df_all = await asyncio.to_thread(load_players_df)
    df_inj = await asyncio.to_thread(load_injured_with_stats)
    all_players = _df_to_records(df_all)
    injured = _df_to_records(df_inj)
    for row in injured:
        row["position"] = get_position_display(str(row.get("position") or ""))
    return templates.TemplateResponse(
        "players.html",
        {
            "request": request,
            "players": all_players,
            "injured": injured,
            "injured_count": len(injured),
        },
    )


@app.get("/schedule", response_class=HTMLResponse)
async def schedule(request: Request) -> HTMLResponse:
    lookup = _team_lookup()
    # Reverse lookup: abbr -> team_id
    abbr_to_id = {abbr: tid for tid, abbr in lookup.items()}
    try:
        df = await asyncio.to_thread(sync_schedule, include_future_days=14)
    except Exception as exc:
        return templates.TemplateResponse(
            "schedule.html",
            {"request": request, "rows": [], "error": str(exc)},
        )
    if df.empty:
        rows: List[dict] = []
    else:
        df = df.copy()
        df["team"] = df["team_id"].map(lookup)
        df["opponent"] = df["opponent_abbr"]
        df["opponent_id"] = df["opponent_abbr"].map(abbr_to_id)
        # Calculate home/away team IDs for links
        df["home_team_id"] = df.apply(
            lambda r: int(r["team_id"]) if r["is_home"] else int(r["opponent_id"]) if pd.notna(r["opponent_id"]) else None,
            axis=1
        )
        df["away_team_id"] = df.apply(
            lambda r: int(r["opponent_id"]) if r["is_home"] and pd.notna(r["opponent_id"]) else int(r["team_id"]),
            axis=1
        )
        df = df[["game_date", "team", "opponent", "is_home", "home_team_id", "away_team_id"]]
        rows = _df_to_records(df)
    return templates.TemplateResponse(
        "schedule.html",
        {"request": request, "rows": rows, "error": None},
    )


@app.get("/matchups", response_class=HTMLResponse)
async def matchups(request: Request, home_team_id: int | None = None, away_team_id: int | None = None) -> HTMLResponse:
    teams = _team_list()
    games = await asyncio.to_thread(get_scheduled_games, 14)
    selected_game = None
    if games:
        selected_game = games[0]
    prediction = None
    error: Optional[str] = None

    if home_team_id and away_team_id:
        home_players = _team_players(home_team_id)
        away_players = _team_players(away_team_id)
        try:
            prediction = predict_matchup(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_players=home_players,
                away_players=away_players,
            )
        except Exception as exc:  # pragma: no cover - analytics exceptions
            error = str(exc)
    return templates.TemplateResponse(
        "matchups.html",
        {
            "request": request,
            "teams": teams,
            "games": games,
            "selected_game": selected_game,
            "prediction": prediction,
            "error": error,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
        },
    )


@app.get("/accuracy", response_class=HTMLResponse)
async def accuracy(
    request: Request,
    run: str | None = None,
    home_team_id: int | None = None,
    away_team_id: int | None = None,
    use_injuries: str | None = None,
) -> HTMLResponse:
    teams = _team_list()
    progress: List[str] = []

    use_injuries_flag = True
    if use_injuries is not None:
        use_injuries_flag = str(use_injuries).lower() not in {"0", "false", "off"}

    results: BacktestResults | None = None
    error = None

    # Only run backtest if explicitly requested via the run parameter
    if run == "1":
        def _cb(msg: str) -> None:
            progress.append(msg)

        try:
            results = await asyncio.to_thread(
                run_backtest,
                5,
                home_team_id,
                away_team_id,
                _cb,
                use_injuries_flag,
            )
        except Exception as exc:  # pragma: no cover - heavy computation
            error = str(exc)

    return templates.TemplateResponse(
        "accuracy.html",
        {
            "request": request,
            "teams": teams,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "results": results,
            "progress": progress,
            "error": error,
            "use_injuries": use_injuries_flag,
        },
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request, status: str | None = None) -> HTMLResponse:
    return templates.TemplateResponse(
        "admin.html",
        {"request": request, "status": status},
    )


@app.post("/admin/reset", response_class=HTMLResponse)
async def admin_reset(request: Request) -> HTMLResponse:
    try:
        if DB_PATH.exists():
            DB_PATH.unlink()
        migrations.init_db()
        status = "Database reset and reinitialized. Run Sync Data next."
    except Exception as exc:  # pragma: no cover - defensive
        status = f"Reset failed: {exc}"
    return await admin(request, status=status)


def create_app() -> FastAPI:
    """Factory for embedding or testing."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=False)
