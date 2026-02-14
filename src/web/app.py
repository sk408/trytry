from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.analytics.autotune import (
    autotune_all,
    autotune_team,
    clear_tuning,
    get_all_tunings,
)
from src.analytics.backtester import BacktestResults, run_backtest, get_actual_game_results
from src.analytics.weight_optimizer import (
    run_weight_optimiser,
    run_per_team_refinement,
    run_combo_optimiser,
    build_residual_calibration,
    load_residual_calibration,
    run_feature_importance,
    run_ml_feature_importance,
    run_fft_error_analysis,
)
from src.analytics.pipeline import run_full_pipeline
from src.analytics.weight_config import (
    get_weight_config,
    clear_weights,
    clear_team_weights,
    load_weights,
)
from src.analytics.live_prediction import LivePrediction, live_predict
from src.analytics.live_recommendations import build_live_recommendations
from src.analytics.prediction import predict_matchup
from src.analytics.stats_engine import get_scheduled_games, get_team_matchup_stats, TeamMatchupStats
from src.data.sync_service import (
    SyncCancelled,
    full_sync,
    sync_injuries,
    sync_injury_history,
    sync_live_scores,
    sync_player_impact,
    sync_schedule,
    sync_team_metrics,
)
from src.data.nba_fetcher import get_current_season
from src.data.gamecast import (
    get_live_games,
    get_game_odds,
    get_box_score,
    get_play_by_play,
    get_game_leaders,
    compute_bonus_status,
    GamecastClient,
    GameInfo,
    GameOdds,
    BoxScore,
    PlayEvent,
    GameLeaders,
)
from src.analytics.prediction import predict_matchup
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


# ── Jinja2 globals: image URL builders ──────────────────────────────
_NBA_TO_ESPN_ABBR_LOGO = {
    "GSW": "gs", "SAS": "sa", "NYK": "ny",
    "NOP": "no", "UTA": "utah", "WAS": "wsh",
}


def _team_logo_url(abbr: str) -> str:
    """Return ESPN CDN URL for a team logo given an NBA abbreviation."""
    espn = _NBA_TO_ESPN_ABBR_LOGO.get(abbr, abbr).lower()
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{espn}.png"


def _player_photo_url(player_id) -> str:
    """Return NBA CDN URL for a player headshot."""
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"


templates.env.globals["team_logo_url"] = _team_logo_url
templates.env.globals["player_photo_url"] = _player_photo_url


@app.on_event("startup")
def startup() -> None:
    migrations.init_db()
    # Start background injury monitor
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(_background_injury_monitor())


async def _background_injury_monitor() -> None:
    """Poll injuries every 5 minutes in the background."""
    import asyncio
    await asyncio.sleep(30)  # initial delay
    from src.notifications.injury_monitor import get_injury_monitor
    monitor = get_injury_monitor()
    while True:
        try:
            await asyncio.get_event_loop().run_in_executor(None, monitor.check)
        except Exception:
            pass
        await asyncio.sleep(300)  # 5 minutes


def _df_to_records(df) -> List[dict]:
    if df is None:
        return []
    return [dict(row._asdict()) for row in df.itertuples(index=False)]


def _team_lookup() -> dict[int, str]:
    from src.analytics.cache import team_cache
    return team_cache.id_to_abbr()


def _team_list() -> List[dict]:
    from src.analytics.cache import team_cache
    return team_cache.team_list()


def _team_players(team_id: int) -> tuple[List[int], dict[int, float]]:
    """Return (player_ids, player_weights) using play_probability >= 0.3."""
    PLAY_PROB_THRESHOLD = 0.3
    pids: List[int] = []
    pw: dict[int, float] = {}
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT player_id, is_injured, injury_note FROM players WHERE team_id = ?",
            (team_id,),
        ).fetchall()
        for r in rows:
            pid, is_injured = int(r[0]), r[1]
            injury_note = r[2] if len(r) > 2 else None
            if not is_injured:
                pids.append(pid)
                pw[pid] = 1.0
            else:
                try:
                    from src.analytics.injury_intelligence import compute_play_probability
                    from src.data.sync_service import _normalise_status_level, _extract_injury_keyword
                    note = injury_note or ""
                    status_raw = note.split(":")[0].strip() if ":" in note else note
                    injury_text = note.split(":", 1)[1].strip() if ":" in note else note
                    if "(" in injury_text:
                        injury_text = injury_text[:injury_text.rfind("(")].strip()
                    status_level = _normalise_status_level(status_raw)
                    keyword = _extract_injury_keyword(injury_text)
                    prob_result = compute_play_probability(pid, "", status_level, keyword, conn)
                    pp = prob_result.composite_probability
                except Exception:
                    pp = 0.0
                if pp >= PLAY_PROB_THRESHOLD:
                    pids.append(pid)
                    pw[pid] = pp
    return pids, pw


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


def _dashboard_stats() -> dict:
    """Get DB counts for dashboard stat cards."""
    try:
        with get_conn() as conn:
            teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
            players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
            logs_count = conn.execute("SELECT COUNT(*) FROM player_stats").fetchone()[0]
            injured = conn.execute(
                "SELECT COUNT(*) FROM players WHERE is_injured = 1"
            ).fetchone()[0]
        return {"teams": teams, "players": players, "game_logs": logs_count, "injured": injured}
    except Exception:
        return {"teams": 0, "players": 0, "game_logs": 0, "injured": 0}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, status: str | None = None, logs: List[str] | None = None) -> HTMLResponse:
    stats = _dashboard_stats()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "status": status or "Ready", "logs": logs or [], "stats": stats},
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


@app.post("/dashboard/team-metrics", response_class=HTMLResponse)
async def dashboard_team_metrics(request: Request) -> HTMLResponse:
    status, logs = await asyncio.to_thread(_run_with_progress, sync_team_metrics, progress_label="Team metrics sync")
    return await dashboard(request, status=status, logs=logs)


@app.post("/dashboard/player-impact", response_class=HTMLResponse)
async def dashboard_player_impact(request: Request) -> HTMLResponse:
    status, logs = await asyncio.to_thread(_run_with_progress, sync_player_impact, progress_label="Player impact sync")
    return await dashboard(request, status=status, logs=logs)


# --- Sync cancel flag (shared across requests) ---
_sync_cancel_flag = threading.Event()


# --- Server-Sent Events (SSE) for real-time progress ---

async def _stream_sync(sync_func, label: str) -> AsyncGenerator[str, None]:
    """Run a sync function and stream progress via SSE.

    Automatically passes cancel_check to sync functions that accept it.
    """
    _sync_cancel_flag.clear()
    msg_queue: queue.Queue[str | None] = queue.Queue()

    def progress_cb(msg: str) -> None:
        msg_queue.put(msg)

    def run_sync() -> None:
        try:
            try:
                sync_func(progress_cb=progress_cb,
                          cancel_check=_sync_cancel_flag.is_set)
            except TypeError:
                # Fallback for sync functions that don't accept cancel_check
                try:
                    sync_func(progress_cb=progress_cb)
                except TypeError:
                    sync_func()
            msg_queue.put(f"[DONE] {label} complete")
        except SyncCancelled:
            msg_queue.put(f"[CANCELLED] {label} stopped — data synced so far is saved")
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


@app.post("/api/sync/cancel")
async def api_sync_cancel() -> dict:
    """Signal any running sync operation to stop after the current step."""
    _sync_cancel_flag.set()
    return {"status": "cancel_requested"}


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


@app.get("/api/sync/team-metrics")
async def stream_sync_team_metrics() -> StreamingResponse:
    """Stream team metrics sync progress via SSE."""
    return StreamingResponse(
        _stream_sync(sync_team_metrics, "Team metrics sync"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sync/player-impact")
async def stream_sync_player_impact() -> StreamingResponse:
    """Stream player impact sync progress via SSE."""
    return StreamingResponse(
        _stream_sync(sync_player_impact, "Player impact sync"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sync/images")
async def stream_sync_images() -> StreamingResponse:
    """Stream image sync progress via SSE."""
    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_sync() -> None:
            try:
                from src.data.image_cache import preload_team_logos, preload_player_photos

                progress_cb("Downloading team logos…")
                logos = preload_team_logos(progress_cb=progress_cb)
                progress_cb(f"Team logos done: {logos} new")

                progress_cb("Downloading player photos…")
                photos = preload_player_photos(progress_cb=progress_cb)
                progress_cb(f"Player photos done: {photos} new")

                msg_queue.put(f"[DONE] Image sync complete ({logos} logos, {photos} photos)")
            except Exception as exc:
                msg_queue.put(f"[ERROR] Image sync failed: {exc}")
            finally:
                msg_queue.put(None)

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

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/backtest")
async def stream_backtest(
    home_team_id: str | None = None,
    away_team_id: str | None = None,
    use_injuries: str | None = None,
    use_cache: str | None = None,
    max_workers: int = 4,
) -> StreamingResponse:
    """Stream backtest progress via SSE."""
    from src.analytics.backtester import load_backtest_cache

    # Convert empty strings to None
    _home_tid: int | None = None
    _away_tid: int | None = None
    if home_team_id and home_team_id.strip():
        try:
            _home_tid = int(home_team_id)
        except ValueError:
            pass
    if away_team_id and away_team_id.strip():
        try:
            _away_tid = int(away_team_id)
        except ValueError:
            pass
    home_team_id_int, away_team_id_int = _home_tid, _away_tid  # noqa: F841
    use_injuries_flag = True
    if use_injuries is not None:
        use_injuries_flag = str(use_injuries).lower() not in {"0", "false", "off"}
    want_cache = str(use_cache).lower() in {"1", "true"} if use_cache is not None else False

    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_bt() -> None:
            try:
                # Try cache first if requested
                results = None
                if want_cache:
                    results = load_backtest_cache(
                        home_team_id_int, away_team_id_int,
                        use_injuries_flag, 1440,
                    )
                    if results is not None:
                        msg_queue.put("Loaded results from cache (instant)")

                if results is None:
                    results = run_backtest(
                        5, home_team_id_int, away_team_id_int, progress_cb, use_injuries_flag,
                        max(1, min(16, max_workers)),
                    )
                # Send full results as JSON so the UI can build tables
                preds = _format_predictions(results) if results.predictions else []
                avg_spread_err = (
                    sum(abs(p.spread_error) for p in results.predictions) / len(results.predictions)
                    if results.predictions else 0.0
                )
                avg_total_err = (
                    sum(abs(p.total_error) for p in results.predictions) / len(results.predictions)
                    if results.predictions else 0.0
                )
                teams_data = []
                for ta in sorted(results.team_accuracy.values(),
                                 key=lambda x: x.spread_accuracy, reverse=True):
                    if ta.games_analyzed > 0:
                        teams_data.append({
                            "team_abbr": ta.team_abbr,
                            "record": f"{ta.wins}-{ta.losses}",
                            "games": ta.games_analyzed,
                            "spread_acc": round(ta.spread_accuracy, 1),
                            "avg_spread_err": round(ta.avg_spread_error, 1),
                            "total_acc": round(ta.total_accuracy, 1),
                            "avg_total_err": round(ta.avg_total_error, 1),
                        })
                payload = json.dumps({
                    "total_games": results.total_games,
                    "spread_acc": round(results.overall_spread_accuracy, 1),
                    "total_acc": round(results.overall_total_accuracy, 1),
                    "avg_spread_err": round(avg_spread_err, 1),
                    "avg_total_err": round(avg_total_err, 1),
                    "teams": teams_data,
                    "predictions": preds,
                })
                msg_queue.put(f"[RESULTS_JSON] {payload}")
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
    from datetime import datetime as _dt

    error = None
    await asyncio.to_thread(sync_live_scores)
    try:
        recs = build_live_recommendations()
    except Exception as exc:  # pragma: no cover - network dependent
        recs = []
        error = str(exc)
    return templates.TemplateResponse(
        "live.html",
        {
            "request": request,
            "recs": recs,
            "error": error,
            "updated_at": _dt.now().strftime("%I:%M:%S %p"),
        },
    )


@app.get("/players", response_class=HTMLResponse)
async def players(request: Request) -> HTMLResponse:
    df_all = await asyncio.to_thread(load_players_df)
    df_inj = await asyncio.to_thread(load_injured_with_stats)
    all_players = _df_to_records(df_all)
    injured = _df_to_records(df_inj)
    for row in injured:
        row["position"] = get_position_display(str(row.get("position") or ""))
    
    # Load manual injuries
    from src.data.injury_scraper import load_manual_injuries
    manual_injuries = load_manual_injuries()
    
    return templates.TemplateResponse(
        "players.html",
        {
            "request": request,
            "players": all_players,
            "injured": injured,
            "injured_count": len(injured),
            "manual_injuries": manual_injuries,
        },
    )


@app.post("/players/injury/add", response_class=HTMLResponse)
async def add_manual_injury(
    request: Request,
    player: str = Form(...),
    team: str = Form(...),
    status: str = Form("Out"),
    injury: str = Form(""),
    position: str = Form(""),
) -> HTMLResponse:
    """Add a manual injury entry."""
    from src.data.injury_scraper import save_manual_injury
    
    success = save_manual_injury(
        player=player.strip(),
        team=team.strip(),
        status=status,
        injury=injury.strip(),
        position=position.strip(),
    )
    
    if success:
        # Trigger injury sync to update database
        await asyncio.to_thread(sync_injuries)
    
    return RedirectResponse(url="/players", status_code=303)


@app.post("/players/injury/remove", response_class=HTMLResponse)
async def remove_manual_injury_endpoint(
    request: Request,
    player: str = Form(...),
    team: str = Form(...),
) -> HTMLResponse:
    """Remove a manual injury entry."""
    from src.data.injury_scraper import remove_manual_injury
    
    remove_manual_injury(player=player.strip(), team=team.strip())
    
    # Trigger injury sync to update database
    await asyncio.to_thread(sync_injuries)
    
    return RedirectResponse(url="/players", status_code=303)


def _format_relative_date(game_date, today) -> str:
    """Format date as Today, Tomorrow, Yesterday, or +/-X days."""
    if game_date == today:
        return "Today"
    
    days_diff = (game_date - today).days
    if days_diff == 1:
        return "Tomorrow"
    elif days_diff > 0:
        return f"+{days_diff} days"
    elif days_diff == -1:
        return "Yesterday"
    else:
        return f"{abs(days_diff)}d ago"


def _format_time_short(time_str: str) -> str:
    """Format time without timezone (e.g., '7:30 PM' instead of '7:30 PM PST')."""
    if not time_str:
        return "TBD"
    # Remove timezone suffix (PST, PDT, EST, EDT, etc.)
    parts = time_str.split()
    if len(parts) >= 2 and parts[-1] in ("PST", "PDT", "EST", "EDT", "MST", "MDT", "CST", "CDT"):
        return " ".join(parts[:-1])
    return time_str


def _utc_to_pacific_short(iso_utc: str) -> str:
    """Convert an ESPN UTC ISO-8601 timestamp to a short Pacific time string.

    ``"2026-02-12T00:30Z"`` → ``"4:30 PM"``
    Falls back to the raw string on parse errors.
    """
    if not iso_utc:
        return "TBD"
    try:
        from datetime import datetime, timezone
        cleaned = iso_utc.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            pacific = dt.astimezone(ZoneInfo("America/Los_Angeles"))
        except (ImportError, KeyError):
            from datetime import timedelta
            utc_dt = dt.astimezone(timezone.utc)
            off = -7 if 3 <= utc_dt.month <= 11 else -8
            pacific = utc_dt + timedelta(hours=off)
        return pacific.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return iso_utc[:16] if len(iso_utc) >= 16 else iso_utc


@app.get("/schedule", response_class=HTMLResponse)
async def schedule(request: Request) -> HTMLResponse:
    from datetime import date as date_type
    
    try:
        df = await asyncio.to_thread(sync_schedule, include_future_days=14)
    except Exception as exc:
        return templates.TemplateResponse(
            "schedule.html",
            {"request": request, "rows": [], "error": str(exc), "season": ""},
        )
    if df.empty:
        rows: List[dict] = []
    else:
        df = df.copy()
        
        # Filter to today and future games only
        today = date_type.today()
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df = df[df["game_date"] >= today]
        
        # Sort by date ascending (today first), then by time
        df = df.sort_values(["game_date", "game_time"] if "game_time" in df.columns else ["game_date"])
        
        # Build display rows with separate abbr for logos + highlighting flags
        rows = []
        for _, r in df.iterrows():
            game_time = r.get("game_time", "") or ""
            arena = r.get("arena", "") or ""
            game_date = r["game_date"]
            date_label = _format_relative_date(game_date, today)
            home_abbr = r.get("home_abbr", "")
            away_abbr = r.get("away_abbr", "")
            
            delta = (game_date - today).days
            
            rows.append({
                "date_label": date_label,
                "game_date_fmt": game_date.strftime("%a %m/%d") if hasattr(game_date, "strftime") else str(game_date),
                "away_abbr": away_abbr,
                "home_abbr": home_abbr,
                "away_team": r.get("away_name") or away_abbr,
                "home_team": r.get("home_name") or home_abbr,
                "game_time": _format_time_short(game_time),
                "venue": arena[:30] + "..." if len(arena) > 30 else arena,
                "home_team_id": int(r["home_team_id"]),
                "away_team_id": int(r["away_team_id"]),
                "is_today": delta == 0,
                "is_tomorrow": delta == 1,
            })
    
    season = get_current_season()
    return templates.TemplateResponse(
        "schedule.html",
        {"request": request, "rows": rows, "error": None, "season": season},
    )


HOME_COURT_ADV = 3.0  # points


def _get_matchup_backtest(home_id: int, away_id: int) -> dict:
    """Get historical performance data for matchup analysis."""
    try:
        games = get_actual_game_results()
    except Exception:
        games = pd.DataFrame()

    if games.empty:
        return {
            "home_abbr": "HOME", "away_abbr": "AWAY",
            "home_record": {"wins": 0, "losses": 0, "avg_pts": 0.0},
            "away_record": {"wins": 0, "losses": 0, "avg_pts": 0.0},
            "h2h_games": [],
        }

    with get_conn() as conn:
        teams_df = pd.read_sql(
            "SELECT team_id, abbreviation FROM teams WHERE team_id IN (?, ?)",
            conn, params=[home_id, away_id]
        )
    abbrs = {int(r["team_id"]): r["abbreviation"] for _, r in teams_df.iterrows()}
    home_abbr = abbrs.get(home_id, "HOME")
    away_abbr = abbrs.get(away_id, "AWAY")

    home_home = games[games["home_team_id"] == home_id]
    away_road = games[games["away_team_id"] == away_id]

    # Home record at home
    hw = sum(1 for _, g in home_home.iterrows() if g["home_score"] > g["away_score"])
    hl = sum(1 for _, g in home_home.iterrows() if g["away_score"] > g["home_score"])
    home_avg = float(home_home["home_score"].mean()) if not home_home.empty else 0.0

    # Away record on road
    aw = sum(1 for _, g in away_road.iterrows() if g["away_score"] > g["home_score"])
    al = sum(1 for _, g in away_road.iterrows() if g["home_score"] > g["away_score"])
    away_avg = float(away_road["away_score"].mean()) if not away_road.empty else 0.0

    # Head to head games
    h2h = []
    for _, g in home_home.iterrows():
        if int(g["away_team_id"]) == away_id:
            h2h.append({
                "date": str(g["game_date"]),
                "home_score": int(g["home_score"]),
                "away_score": int(g["away_score"]),
                "winner": home_abbr if g["home_score"] > g["away_score"] else away_abbr,
            })

    return {
        "home_abbr": home_abbr,
        "away_abbr": away_abbr,
        "home_record": {"wins": hw, "losses": hl, "avg_pts": home_avg},
        "away_record": {"wins": aw, "losses": al, "avg_pts": away_avg},
        "h2h_games": h2h,
    }


def _get_injury_summary(stats: TeamMatchupStats) -> dict:
    """Get injury impact summary for a team."""
    injured = [p for p in stats.players if p.is_injured and p.mpg > 0]
    if not injured:
        return {"status": "healthy", "text": "No injuries", "lost_ppg": 0}

    lost_ppg = sum(p.ppg for p in injured)
    key = [p for p in injured if p.mpg >= 25]
    rotation = [p for p in injured if 15 <= p.mpg < 25]

    if key:
        names = ", ".join(p.name.split()[-1] for p in key[:2])
        return {"status": "critical", "text": f"KEY OUT: {names}", "lost_ppg": lost_ppg}
    elif rotation:
        names = ", ".join(p.name.split()[-1] for p in rotation[:2])
        return {"status": "moderate", "text": f"OUT: {names}", "lost_ppg": lost_ppg}
    else:
        return {"status": "minor", "text": f"{len(injured)} minor injuries", "lost_ppg": lost_ppg}


def _players_to_dicts(stats: TeamMatchupStats, opp_id: int, is_home: bool) -> List[dict]:
    """Convert player stats to template-friendly dicts."""
    result = []
    players = [p for p in stats.players if p.mpg > 0][:12]
    for p in players:
        base = p.ppg * 0.4
        loc = (p.ppg_home if is_home else p.ppg_away) * 0.3
        vs = (p.ppg_vs_opp if p.games_vs_opp > 0 else p.ppg) * 0.3
        proj = base + loc + vs
        result.append({
            "player_id": p.player_id,
            "name": p.name,
            "position": p.position,
            "ppg": p.ppg,
            "rpg": p.rpg,
            "apg": p.apg,
            "mpg": p.mpg,
            "ppg_home": p.ppg_home,
            "ppg_away": p.ppg_away,
            "ppg_vs_opp": p.ppg_vs_opp if p.games_vs_opp > 0 else None,
            "projected": proj,
            "is_injured": p.is_injured,
            "play_probability": getattr(p, "play_probability", 1.0),
            "injury_status": getattr(p, "injury_status", ""),
            "injury_keyword": getattr(p, "injury_keyword", ""),
        })
    return result


@app.get("/matchups", response_class=HTMLResponse)
async def matchups(request: Request, home_team_id: int | None = None, away_team_id: int | None = None) -> HTMLResponse:
    from datetime import datetime, date as date_type
    
    teams = _team_list()
    today = date_type.today()

    # ── Upcoming games: fetch schedule directly and filter to today+ ──
    upcoming_games: list[dict] = []
    try:
        sched_df = await asyncio.to_thread(sync_schedule, include_future_days=14)
        if not sched_df.empty:
            sched_df = sched_df.copy()
            sched_df["game_date"] = pd.to_datetime(sched_df["game_date"]).dt.date
            sched_df = sched_df[sched_df["game_date"] >= today]
            sort_cols = ["game_date", "game_time"] if "game_time" in sched_df.columns else ["game_date"]
            sched_df = sched_df.sort_values(sort_cols)
            # Deduplicate by date + teams
            seen: set[tuple] = set()
            for _, r in sched_df.iterrows():
                h_id = int(r["home_team_id"])
                a_id = int(r["away_team_id"])
                key = (r["game_date"], min(h_id, a_id), max(h_id, a_id))
                if key in seen:
                    continue
                seen.add(key)
                game_time = str(r.get("game_time", "") or "")
                upcoming_games.append({
                    "game_date": r["game_date"],
                    "date_label": _format_relative_date(r["game_date"], today),
                    "game_date_fmt": r["game_date"].strftime("%a %m/%d"),
                    "home_team_id": h_id,
                    "away_team_id": a_id,
                    "home_abbr": r.get("home_abbr", ""),
                    "away_abbr": r.get("away_abbr", ""),
                    "game_time": _format_time_short(game_time),
                })
    except Exception:
        pass

    # ── Past results: actual scores from player_stats (most recent first) ──
    past_games: list[dict] = []
    try:
        results_df = await asyncio.to_thread(get_actual_game_results)
        if not results_df.empty:
            results_df = results_df.sort_values("game_date", ascending=False).head(60)
            for _, row in results_df.iterrows():
                gd_str = str(row["game_date"])
                try:
                    gd = date_type.fromisoformat(gd_str[:10])
                except ValueError:
                    gd = None
                home_score = int(row["home_score"])
                away_score = int(row["away_score"])
                winner_abbr = row["home_abbr"] if home_score > away_score else row["away_abbr"]
                past_games.append({
                    "game_date": gd_str,
                    "game_date_fmt": gd.strftime("%a %m/%d") if gd else gd_str,
                    "date_label": _format_relative_date(gd, today) if gd else gd_str,
                    "home_team_id": int(row["home_team_id"]),
                    "away_team_id": int(row["away_team_id"]),
                    "home_abbr": row["home_abbr"],
                    "away_abbr": row["away_abbr"],
                    "home_score": home_score,
                    "away_score": away_score,
                    "total": home_score + away_score,
                    "winner": winner_abbr,
                })
    except Exception:
        pass

    prediction = None
    home_stats = None
    away_stats = None
    backtest = None
    home_injury = None
    away_injury = None
    home_players = []
    away_players = []
    error: Optional[str] = None
    game_start_iso: Optional[str] = None  # ISO timestamp for countdown

    if home_team_id and away_team_id:
        # Try to find the game time from schedule
        try:
            schedule_df = await asyncio.to_thread(sync_schedule, include_future_days=14)
            if not schedule_df.empty:
                # Find game matching these teams
                for _, row in schedule_df.iterrows():
                    h_id = int(row.get("home_team_id", 0))
                    a_id = int(row.get("away_team_id", 0))
                    if h_id == home_team_id and a_id == away_team_id:
                        game_date = row.get("game_date")
                        game_time = row.get("game_time", "")
                        if game_date:
                            # Parse date
                            if hasattr(game_date, 'isoformat'):
                                gd = game_date
                            else:
                                gd = pd.to_datetime(game_date).date()
                            
                            # Parse time and build ISO timestamp
                            if game_time and game_time != "TBD":
                                try:
                                    # Remove timezone suffix and parse
                                    time_clean = game_time.split()[0:2]  # "7:30 PM"
                                    time_str = " ".join(time_clean)
                                    dt = datetime.strptime(f"{gd} {time_str}", "%Y-%m-%d %I:%M %p")
                                    game_start_iso = dt.isoformat()
                                except (ValueError, IndexError):
                                    # Just use noon if can't parse time
                                    game_start_iso = f"{gd}T12:00:00"
                            else:
                                # Default to noon if no time
                                game_start_iso = f"{gd}T12:00:00"
                        break
        except Exception:
            pass  # Don't fail if schedule lookup fails
        
        try:
            # Get comprehensive team stats
            home_stats = await asyncio.to_thread(
                get_team_matchup_stats, home_team_id, away_team_id, True
            )
            away_stats = await asyncio.to_thread(
                get_team_matchup_stats, away_team_id, home_team_id, False
            )

            if home_stats.players and away_stats.players:
                # Use play_probability for roster selection (>= 0.3)
                PLAY_PROB_THRESHOLD = 0.3
                home_player_ids = []
                home_pw: dict[int, float] = {}
                for p in home_stats.players:
                    pp = getattr(p, "play_probability", 0.0 if p.is_injured else 1.0)
                    if pp >= PLAY_PROB_THRESHOLD:
                        home_player_ids.append(p.player_id)
                        home_pw[p.player_id] = pp
                away_player_ids = []
                away_pw: dict[int, float] = {}
                for p in away_stats.players:
                    pp = getattr(p, "play_probability", 0.0 if p.is_injured else 1.0)
                    if pp >= PLAY_PROB_THRESHOLD:
                        away_player_ids.append(p.player_id)
                        away_pw[p.player_id] = pp
                full_pred = await asyncio.to_thread(
                    predict_matchup,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    home_players=home_player_ids,
                    away_players=away_player_ids,
                    home_player_weights=home_pw,
                    away_player_weights=away_pw,
                )
                prediction = {
                    "predicted_spread": full_pred.predicted_spread,
                    "predicted_total": full_pred.predicted_total,
                    "home_projected": full_pred.predicted_home_score,
                    "away_projected": full_pred.predicted_away_score,
                    "four_factors_adj": full_pred.four_factors_adj,
                    "clutch_adj": full_pred.clutch_adj,
                    "fatigue_adj": full_pred.fatigue_adj,
                }

                # Get injury summaries
                home_injury = _get_injury_summary(home_stats)
                away_injury = _get_injury_summary(away_stats)

                # Get player breakdowns
                home_players = _players_to_dicts(home_stats, away_team_id, True)
                away_players = _players_to_dicts(away_stats, home_team_id, False)

                # Get historical backtest data
                backtest = await asyncio.to_thread(_get_matchup_backtest, home_team_id, away_team_id)
            else:
                error = "No player data. Run sync first."
        except Exception as exc:
            error = str(exc)

    return templates.TemplateResponse(
        "matchups.html",
        {
            "request": request,
            "teams": teams,
            "upcoming_games": upcoming_games,
            "past_games": past_games,
            "prediction": prediction,
            "home_stats": home_stats,
            "away_stats": away_stats,
            "backtest": backtest,
            "home_injury": home_injury,
            "away_injury": away_injury,
            "home_players": home_players,
            "away_players": away_players,
            "error": error,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "game_start_iso": game_start_iso,
        },
    )


def _format_predictions(results: BacktestResults) -> List[dict]:
    """Format predictions for template display."""
    preds = []
    for p in sorted(results.predictions, key=lambda x: str(x.game_date), reverse=True)[:50]:
        # Format predicted/actual winner
        pred_winner = p.home_abbr if p.predicted_winner == "HOME" else (
            p.away_abbr if p.predicted_winner == "AWAY" else "Close"
        )
        actual_winner = p.home_abbr if p.actual_winner == "HOME" else (
            p.away_abbr if p.actual_winner == "AWAY" else "Tie"
        )
        # Format injuries (last names only)
        home_inj = ", ".join(n.split()[-1] for n in p.home_injuries[:2]) if p.home_injuries else ""
        away_inj = ", ".join(n.split()[-1] for n in p.away_injuries[:2]) if p.away_injuries else ""
        injury_text = ""
        if home_inj:
            injury_text = f"{p.home_abbr}: {home_inj}"
        if away_inj:
            injury_text += (" | " if injury_text else "") + f"{p.away_abbr}: {away_inj}"
        
        preds.append({
            "game_date": str(p.game_date),
            "matchup": f"{p.away_abbr} @ {p.home_abbr}",
            "final_score": f"{int(p.actual_away_score)}-{int(p.actual_home_score)}",
            "pred_winner": pred_winner,
            "actual_winner": actual_winner,
            "winner_correct": p.winner_correct,
            "pred_score": f"{int(p.predicted_away_score)}-{int(p.predicted_home_score)}",
            "score_diff": f"H:{p.home_score_error:+.0f} A:{p.away_score_error:+.0f}",
            "pred_total": int(p.predicted_total),
            "actual_total": int(p.actual_total),
            "total_diff": p.total_error,
            "injuries": injury_text or "-",
            "has_injuries": bool(home_inj or away_inj),
        })
    return preds


@app.get("/api/backtest-cache-age")
async def backtest_cache_age(
    home_team_id: str | None = None,
    away_team_id: str | None = None,
    use_injuries: str | None = None,
) -> dict:
    """Return the age (in minutes) of the cached backtest, or null."""
    from src.analytics.backtester import get_backtest_cache_age

    home_tid: int | None = None
    away_tid: int | None = None
    if home_team_id and home_team_id.strip():
        try:
            home_tid = int(home_team_id)
        except ValueError:
            pass
    if away_team_id and away_team_id.strip():
        try:
            away_tid = int(away_team_id)
        except ValueError:
            pass
    inj_flag = True
    if use_injuries is not None:
        inj_flag = str(use_injuries).lower() not in {"0", "false", "off"}

    age = get_backtest_cache_age(home_tid, away_tid, inj_flag)
    return {"age_minutes": age}


@app.get("/accuracy", response_class=HTMLResponse)
async def accuracy(request: Request) -> HTMLResponse:
    """Serve the accuracy page.  Backtest runs via SSE (/api/backtest)."""
    teams = _team_list()
    return templates.TemplateResponse(
        "accuracy.html",
        {
            "request": request,
            "teams": teams,
        },
    )


# --- Model optimisation SSE endpoints ---


@app.get("/api/optimize")
async def stream_optimize(trials: int = 200) -> StreamingResponse:
    """Stream weight optimisation progress via SSE."""
    n_trials = max(50, min(2000, trials))

    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_opt() -> None:
            try:
                result = run_weight_optimiser(n_trials=n_trials, progress_cb=progress_cb)
                msg_queue.put(
                    f"[RESULT] baseline_loss={result.baseline_loss:.2f},"
                    f"best_loss={result.best_loss:.2f},"
                    f"improvement={result.improvement_pct:+.1f}%,"
                    f"trials={result.trials_run}"
                )
                msg_queue.put("[DONE] Weight optimisation complete")
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_opt, daemon=True)
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


@app.get("/api/calibrate")
async def stream_calibrate() -> StreamingResponse:
    """Stream residual calibration progress via SSE."""
    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_cal() -> None:
            try:
                calibration = build_residual_calibration(progress_cb=progress_cb)
                for label, data in calibration.items():
                    msg_queue.put(
                        f"[RESULT] bin={label},"
                        f"range=[{data['bin_low']:+.0f},{data['bin_high']:+.0f}),"
                        f"residual={data['avg_residual']:+.3f},"
                        f"n={data['sample_count']}"
                    )
                msg_queue.put("[DONE] Calibration complete")
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_cal, daemon=True)
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


@app.get("/api/feature-importance")
async def stream_feature_importance() -> StreamingResponse:
    """Stream feature importance analysis progress via SSE."""
    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_fi() -> None:
            try:
                results = run_feature_importance(progress_cb=progress_cb)
                for f in results:
                    verdict = "HELPS" if f.impact > 0.05 else ("HURTS" if f.impact < -0.05 else "neutral")
                    msg_queue.put(
                        f"[RESULT] feature={f.feature_name},"
                        f"impact={f.impact:+.3f},"
                        f"impact_pct={f.impact_pct:+.2f}%,"
                        f"verdict={verdict}"
                    )
                msg_queue.put("[DONE] Feature importance complete")
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_fi, daemon=True)
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


@app.get("/api/grouped-feature-importance")
async def stream_grouped_feature_importance() -> StreamingResponse:
    """Stream grouped feature importance analysis via SSE."""
    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_gfi() -> None:
            try:
                from src.analytics.weight_optimizer import run_grouped_feature_importance
                results = run_grouped_feature_importance(progress_cb=progress_cb)
                for g in results:
                    verdict = "HELPS" if g.impact > 0.1 else ("HURTS" if g.impact < -0.1 else "neutral")
                    weights_str = " | ".join(g.features_disabled)
                    msg_queue.put(
                        f"[RESULT] group={g.group_name},"
                        f"weights={weights_str},"
                        f"impact={g.impact:+.3f},"
                        f"impact_pct={g.impact_pct:+.2f}%,"
                        f"verdict={verdict}"
                    )
                msg_queue.put("[DONE] Grouped importance complete")
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_gfi, daemon=True)
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


@app.get("/api/ml-feature-importance")
async def stream_ml_feature_importance() -> StreamingResponse:
    """Stream ML (XGBoost + SHAP) feature importance analysis via SSE."""
    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_ml() -> None:
            try:
                results = run_ml_feature_importance(progress_cb=progress_cb)
                for f in results:
                    msg_queue.put(
                        f"[RESULT] feature={f.feature_name},"
                        f"shap={f.shap_importance:.4f},"
                        f"direction={f.direction}"
                    )
                msg_queue.put("[DONE] ML feature importance complete")
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_ml, daemon=True)
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


@app.get("/api/fft-analysis")
async def stream_fft_analysis() -> StreamingResponse:
    """Stream FFT error pattern analysis via SSE."""
    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_fft() -> None:
            try:
                patterns = run_fft_error_analysis(progress_cb=progress_cb)
                for p in patterns:
                    msg_queue.put(
                        f"[RESULT] description={p.description},"
                        f"period_games={p.period_games},"
                        f"period_days={p.period_days},"
                        f"magnitude={p.magnitude}"
                    )
                if not patterns:
                    msg_queue.put("[RESULT] description=No significant patterns detected (good),magnitude=0")
                msg_queue.put("[DONE] FFT analysis complete")
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_fft, daemon=True)
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


@app.get("/api/ml-train")
async def stream_ml_train() -> StreamingResponse:
    """Stream ML ensemble model training via SSE."""
    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_train() -> None:
            try:
                from src.analytics.prediction import precompute_game_data
                from src.analytics.ml_model import train_models, reload_models

                games = precompute_game_data(progress_cb=progress_cb)
                if not games:
                    msg_queue.put("[ERROR] No precomputed games. Run a data sync first.")
                    return
                result = train_models(games, progress_cb=progress_cb)
                reload_models()

                # Send structured results
                msg_queue.put(
                    f"[METRIC] spread_train_mae={result.spread_train_mae:.3f},"
                    f"spread_val_mae={result.spread_val_mae:.3f},"
                    f"total_train_mae={result.total_train_mae:.3f},"
                    f"total_val_mae={result.total_val_mae:.3f},"
                    f"n_train={result.n_train},"
                    f"n_val={result.n_val},"
                    f"n_features={result.n_features}"
                )

                # Send SHAP features
                for name, imp in result.shap_spread_features:
                    msg_queue.put(f"[SHAP_SPREAD] {name}={imp:.4f}")
                for name, imp in result.shap_total_features:
                    msg_queue.put(f"[SHAP_TOTAL] {name}={imp:.4f}")

                # Send gain-based features
                for name, gain in result.top_spread_features:
                    msg_queue.put(f"[GAIN_SPREAD] {name}={gain:.2f}")
                for name, gain in result.top_total_features:
                    msg_queue.put(f"[GAIN_TOTAL] {name}={gain:.2f}")

                msg_queue.put(
                    f"[DONE] ML models trained — "
                    f"spread val MAE={result.spread_val_mae:.2f}, "
                    f"total val MAE={result.total_val_mae:.2f}"
                )
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_train, daemon=True)
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


@app.get("/api/team-refinement")
async def stream_team_refinement(trials: int = 100) -> StreamingResponse:
    """Stream per-team weight refinement progress via SSE."""
    n_trials = max(20, min(1000, trials))

    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_refine() -> None:
            try:
                results = run_per_team_refinement(
                    n_trials=n_trials,
                    progress_cb=progress_cb,
                )
                for r in results:
                    msg_queue.put(
                        f"[RESULT] abbr={r.team_abbr},"
                        f"used_team={str(r.used_team_weights).lower()},"
                        f"global_holdout={r.global_loss_recent:.2f},"
                        f"team_holdout={r.team_loss_recent:.2f},"
                        f"reason={r.reason}"
                    )
                adopted = sum(1 for r in results if r.used_team_weights)
                msg_queue.put(
                    f"[DONE] Per-team refinement complete: "
                    f"{adopted}/{len(results)} teams got custom weights"
                )
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_refine, daemon=True)
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


# --- Cancel flags (shared across requests) ---
_pipeline_cancel_flag = threading.Event()
_continuous_cancel_flag = threading.Event()


@app.get("/api/continuous-optimize")
async def stream_continuous_optimize(trials: int = 200) -> StreamingResponse:
    """Stream continuous optimisation (loops until cancelled) via SSE."""
    n_trials = max(50, min(2000, trials))
    _continuous_cancel_flag.clear()

    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_continuous() -> None:
            try:
                from src.analytics.weight_optimizer import run_continuous_optimiser
                result = run_continuous_optimiser(
                    n_trials=n_trials,
                    team_trials=n_trials,
                    progress_cb=progress_cb,
                    cancel_check=lambda: _continuous_cancel_flag.is_set(),
                )
                msg_queue.put(
                    f"[RESULT] rounds={result.rounds_completed},"
                    f"improvements={result.global_improvements},"
                    f"starting_loss={result.starting_loss:.2f},"
                    f"best_loss={result.best_global_loss:.2f},"
                    f"teams_refined={result.teams_refined}/{result.total_teams},"
                    f"seconds={result.total_seconds:.0f}"
                )
                msg_queue.put("[DONE] Continuous optimisation stopped")
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_continuous, daemon=True)
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


@app.post("/api/continuous-optimize/cancel")
async def api_continuous_cancel() -> dict:
    """Signal the continuous optimiser to stop after the current round."""
    _continuous_cancel_flag.set()
    return {"status": "cancel_requested"}


@app.get("/api/optimize-all")
async def stream_optimize_all(trials: int = 200) -> StreamingResponse:
    """Stream combo optimisation (global + per-team) via SSE."""
    n_trials = max(50, min(2000, trials))

    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_combo() -> None:
            try:
                result = run_combo_optimiser(
                    n_trials=n_trials,
                    team_trials=n_trials,
                    progress_cb=progress_cb,
                )
                gr = result.global_result
                adopted = sum(1 for r in result.team_results if r.used_team_weights)
                msg_queue.put(
                    f"[RESULT] global_baseline={gr.baseline_loss:.2f},"
                    f"global_best={gr.best_loss:.2f},"
                    f"improvement={gr.improvement_pct:+.1f}%,"
                    f"teams_refined={adopted}/{len(result.team_results)},"
                    f"seconds={result.total_seconds:.0f}"
                )
                for r in result.team_results:
                    msg_queue.put(
                        f"[TEAM] abbr={r.team_abbr},"
                        f"used_team={str(r.used_team_weights).lower()},"
                        f"global_holdout={r.global_loss_recent:.2f},"
                        f"team_holdout={r.team_loss_recent:.2f}"
                    )
                msg_queue.put("[DONE] Combo optimisation complete")
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_combo, daemon=True)
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


@app.get("/api/full-pipeline")
async def stream_full_pipeline(
    trials: int = 200,
    max_workers: int = 4,
    force_rerun: str = "0",
) -> StreamingResponse:
    """Stream the full optimisation pipeline via SSE."""
    n_trials = max(50, min(2000, trials))
    n_workers = max(1, min(16, max_workers))
    force = force_rerun.lower() in {"1", "true", "yes"}

    _pipeline_cancel_flag.clear()

    async def generate() -> AsyncGenerator[str, None]:
        msg_queue: queue.Queue[str | None] = queue.Queue()

        def progress_cb(msg: str) -> None:
            msg_queue.put(msg)

        def run_pipe() -> None:
            try:
                summary = run_full_pipeline(
                    n_trials=n_trials,
                    team_trials=n_trials,
                    max_workers=n_workers,
                    progress_cb=progress_cb,
                    cancel_check=_pipeline_cancel_flag.is_set,
                    force_rerun=force,
                )
                total_s = summary.get("total_seconds", 0)
                msg_queue.put(f"[RESULT] total_seconds={total_s:.0f}")
                for name, info in summary.get("steps", {}).items():
                    status = info.get("status", "?")
                    secs = info.get("seconds", 0)
                    msg_queue.put(f"[STEP] name={name},status={status},seconds={secs:.1f}")
                if summary.get("cancelled"):
                    msg_queue.put("[DONE] Pipeline cancelled by user")
                else:
                    msg_queue.put(f"[DONE] Full pipeline complete in {total_s:.0f}s")
            except Exception as exc:
                msg_queue.put(f"[ERROR] {exc}")
            finally:
                msg_queue.put(None)

        thread = threading.Thread(target=run_pipe, daemon=True)
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


@app.post("/api/pipeline/cancel")
async def api_pipeline_cancel() -> dict:
    """Set the cancel flag for the running pipeline."""
    _pipeline_cancel_flag.set()
    return {"status": "ok", "message": "Cancel signal sent"}


@app.post("/api/weights/clear")
async def api_clear_weights() -> dict:
    """Clear optimised weights (global + per-team) and revert to defaults."""
    await asyncio.to_thread(clear_weights)
    await asyncio.to_thread(clear_team_weights)
    return {"status": "ok", "message": "Weights reset to defaults (global + per-team)"}


@app.get("/api/weights")
async def api_get_weights() -> dict:
    """Return current active weights."""
    cfg = get_weight_config(force_reload=True)
    return cfg.to_dict()


@app.get("/api/calibration")
async def api_get_calibration() -> List[dict]:
    """Return saved residual calibration bins."""
    return await asyncio.to_thread(load_residual_calibration)


@app.get("/autotune", response_class=HTMLResponse)
async def autotune_page(
    request: Request,
    run: str | None = None,
    team_id: str | None = None,
    strength: float = 0.75,
    min_threshold: float = 1.5,
) -> HTMLResponse:
    """Autotune page -- run player-level historical analysis per team."""
    # Convert team_id: empty string or None → None, otherwise int
    team_id_int: int | None = None
    if team_id and team_id.strip():
        try:
            team_id_int = int(team_id)
        except ValueError:
            team_id_int = None

    teams = _team_list()
    progress: List[str] = []
    results: List[dict] = []
    error = None

    if run == "1":
        def _cb(msg: str) -> None:
            progress.append(msg)

        try:
            if team_id_int:
                # Single team
                res = await asyncio.to_thread(
                    autotune_team, team_id_int, strength, min_threshold, _cb,
                )
                results = [res]
            else:
                # All teams
                results = await asyncio.to_thread(
                    autotune_all, strength, min_threshold, _cb,
                )
        except Exception as exc:
            error = str(exc)

    # Always load current tunings for display
    current_tunings = await asyncio.to_thread(get_all_tunings)

    return templates.TemplateResponse(
        "autotune.html",
        {
            "request": request,
            "teams": teams,
            "team_id": team_id_int,
            "strength": strength,
            "min_threshold": min_threshold,
            "progress": progress,
            "results": results,
            "current_tunings": current_tunings,
            "error": error,
        },
    )


@app.post("/autotune/clear", response_class=HTMLResponse)
async def autotune_clear(
    request: Request,
    team_id: str | None = Form(None),
) -> HTMLResponse:
    """Clear autotune corrections for one team or all teams."""
    tid: int | None = None
    if team_id and team_id.strip():
        try:
            tid = int(team_id)
        except ValueError:
            pass
    await asyncio.to_thread(clear_tuning, tid)
    return RedirectResponse(url="/autotune", status_code=303)


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request, status: str | None = None) -> HTMLResponse:
    db_path = str(DB_PATH)
    try:
        size_bytes = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        if size_bytes >= 1_048_576:
            db_size = f"{size_bytes / 1_048_576:.1f} MB"
        elif size_bytes >= 1024:
            db_size = f"{size_bytes / 1024:.1f} KB"
        else:
            db_size = f"{size_bytes} bytes"
    except Exception:
        db_size = "unknown"
    return templates.TemplateResponse(
        "admin.html",
        {"request": request, "status": status, "db_path": db_path, "db_size": db_size},
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


# --- Gamecast (ESPN real-time game data) ---

# ESPN uses shorter abbreviations for a few teams.  Map to the NBA-API
# values stored in our ``teams`` table.
_ESPN_TO_NBA_ABBR = {
    "GS": "GSW",
    "SA": "SAS",
    "NY": "NYK",
    "NO": "NOP",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def _get_team_id_by_abbr(abbr: str) -> Optional[int]:
    """Look up team ID by abbreviation.

    Handles known ESPN ↔ NBA-API abbreviation mismatches (e.g. GS → GSW).
    Uses ``team_cache`` — zero DB calls after the first lookup.
    """
    from src.analytics.cache import team_cache
    canonical = _ESPN_TO_NBA_ABBR.get(abbr, abbr)
    return team_cache.get_id(canonical)


def _get_our_prediction(home_abbr: str, away_abbr: str) -> Optional[dict]:
    """Get our model's prediction for the matchup."""
    home_id = _get_team_id_by_abbr(home_abbr)
    away_id = _get_team_id_by_abbr(away_abbr)
    
    if not home_id or not away_id:
        return None
    
    try:
        home_players, home_pw = _team_players(home_id)
        away_players, away_pw = _team_players(away_id)
        
        if not home_players or not away_players:
            return None
        
        prediction = predict_matchup(
            home_team_id=home_id,
            away_team_id=away_id,
            home_players=home_players,
            away_players=away_players,
            home_player_weights=home_pw,
            away_player_weights=away_pw,
        )
        
        # Calculate predicted final scores
        home_proj = (prediction.predicted_total + prediction.predicted_spread) / 2
        away_proj = (prediction.predicted_total - prediction.predicted_spread) / 2
        
        return {
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "predicted_spread": prediction.predicted_spread,
            "predicted_total": prediction.predicted_total,
            "home_projected_score": home_proj,
            "away_projected_score": away_proj,
        }
    except Exception as e:
        print(f"[gamecast] Error getting prediction: {e}")
        return None


@app.get("/gamecast", response_class=HTMLResponse)
async def gamecast(request: Request, game_id: str | None = None) -> HTMLResponse:
    """Gamecast page with live play-by-play and odds."""
    games = await asyncio.to_thread(get_live_games)
    
    selected_game: GameInfo | None = None
    odds: GameOdds | None = None
    box: BoxScore | None = None
    leaders: GameLeaders | None = None
    recent_plays: List[PlayEvent] = []
    all_plays: List[PlayEvent] = []
    our_prediction: Optional[dict] = None
    live_pred: Optional[LivePrediction] = None
    bonus: dict = {}
    
    if game_id:
        # Find the selected game
        selected_game = next((g for g in games if g.game_id == game_id), None)
        
        # Fetch initial data
        odds = await asyncio.to_thread(get_game_odds, game_id)
        box = await asyncio.to_thread(get_box_score, game_id)
        leaders = await asyncio.to_thread(get_game_leaders, game_id)
        all_plays = await asyncio.to_thread(get_play_by_play, game_id, "")
        recent_plays = all_plays[:30]  # Last 30 plays for display

        # Compute bonus status from full play list
        if selected_game and all_plays:
            # all_plays is newest-first; compute_bonus_status iterates all
            bonus = compute_bonus_status(
                all_plays,
                selected_game.home_abbr,
                selected_game.away_abbr,
                selected_game.period,
            )
        
        # Get our model's pre-game prediction (legacy)
        if selected_game:
            our_prediction = await asyncio.to_thread(
                _get_our_prediction, selected_game.home_abbr, selected_game.away_abbr
            )

            # Enhanced live prediction (blended engine) -- only for in-progress games
            if selected_game.status == "in_progress":
                home_id = _get_team_id_by_abbr(selected_game.home_abbr)
                away_id = _get_team_id_by_abbr(selected_game.away_abbr)
                if home_id and away_id:
                    try:
                        live_pred = await asyncio.to_thread(
                            live_predict,
                            home_team_id=home_id,
                            away_team_id=away_id,
                            home_score=selected_game.home_score,
                            away_score=selected_game.away_score,
                            period=selected_game.period,
                            clock=selected_game.clock,
                            game_id=game_id,
                            game_date_str=selected_game.start_time[:10] if selected_game.start_time else "",
                        )
                    except Exception as e:
                        print(f"[gamecast] live_predict error: {e}")
    
    return templates.TemplateResponse(
        "gamecast.html",
        {
            "request": request,
            "games": games,
            "selected_game": selected_game,
            "game_id": game_id,
            "odds": odds,
            "box": box,
            "leaders": leaders,
            "recent_plays": recent_plays,
            "our_prediction": our_prediction,
            "live_pred": live_pred,
            "bonus": bonus,
            "to_pacific": _utc_to_pacific_short,
        },
    )


@app.get("/api/gamecast/games")
async def api_gamecast_games() -> List[dict]:
    """Get list of today's NBA games."""
    games = await asyncio.to_thread(get_live_games)
    return [
        {
            "game_id": g.game_id,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_abbr": g.home_abbr,
            "away_abbr": g.away_abbr,
            "home_score": g.home_score,
            "away_score": g.away_score,
            "status": g.status,
            "period": g.period,
            "clock": g.clock,
            "start_time": g.start_time,
            "start_time_pt": _utc_to_pacific_short(g.start_time),
        }
        for g in games
    ]


@app.get("/api/gamecast/odds/{game_id}")
async def api_gamecast_odds(game_id: str) -> dict:
    """Get current odds for a game."""
    odds = await asyncio.to_thread(get_game_odds, game_id)
    if not odds:
        return {"error": "Odds not available"}
    return {
        "spread": odds.spread,
        "spread_odds": odds.spread_odds,
        "over_under": odds.over_under,
        "over_odds": odds.over_odds,
        "under_odds": odds.under_odds,
        "home_ml": odds.home_ml,
        "away_ml": odds.away_ml,
        "home_win_pct": odds.home_win_pct,
        "away_win_pct": odds.away_win_pct,
        "home_ats_record": odds.home_ats_record,
        "away_ats_record": odds.away_ats_record,
    }


@app.get("/api/gamecast/boxscore/{game_id}")
async def api_gamecast_boxscore(game_id: str) -> dict:
    """Get current box score for a game."""
    box = await asyncio.to_thread(get_box_score, game_id)
    if not box:
        return {"error": "Box score not available"}
    
    def player_to_dict(p):
        return {
            "name": p.name,
            "position": p.position,
            "minutes": p.minutes,
            "points": p.points,
            "rebounds": p.rebounds,
            "assists": p.assists,
            "fg": p.fg,
            "fg3": p.fg3,
            "ft": p.ft,
        }
    
    return {
        "home_players": [player_to_dict(p) for p in box.home_players],
        "away_players": [player_to_dict(p) for p in box.away_players],
        "home_totals": box.home_totals,
        "away_totals": box.away_totals,
    }


@app.get("/api/gamecast/stream/{game_id}")
async def api_gamecast_stream(game_id: str) -> StreamingResponse:
    """Stream live game events via SSE."""
    
    async def generate() -> AsyncGenerator[str, None]:
        client = GamecastClient(game_id)
        last_play_id = ""
        last_odds_time = 0.0
        poll_interval = 10.0
        odds_interval = 30.0
        
        # Send initial odds
        odds = await asyncio.to_thread(get_game_odds, game_id)
        if odds:
            yield f"data: {json.dumps({'type': 'odds', 'data': _odds_to_dict(odds)})}\n\n"
            last_odds_time = time.time()
        
        # Stream play-by-play
        while True:
            try:
                # Check for new plays
                plays = await asyncio.to_thread(get_play_by_play, game_id, last_play_id)
                
                for play in reversed(plays):  # Oldest first
                    yield f"data: {json.dumps({'type': 'play', 'data': _play_to_dict(play)})}\n\n"
                    last_play_id = play.event_id
                
                # Update score
                if plays:
                    games = await asyncio.to_thread(get_live_games)
                    game = next((g for g in games if g.game_id == game_id), None)
                    if game:
                        yield f"data: {json.dumps({'type': 'score', 'data': _game_to_dict(game)})}\n\n"
                
                # Periodic odds update
                now = time.time()
                if now - last_odds_time >= odds_interval:
                    odds = await asyncio.to_thread(get_game_odds, game_id)
                    if odds:
                        yield f"data: {json.dumps({'type': 'odds', 'data': _odds_to_dict(odds)})}\n\n"
                    last_odds_time = now
                
                # Check if game is over
                games = await asyncio.to_thread(get_live_games)
                game = next((g for g in games if g.game_id == game_id), None)
                if game and game.status == "final":
                    yield f"data: {json.dumps({'type': 'final', 'data': _game_to_dict(game)})}\n\n"
                    break
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                await asyncio.sleep(poll_interval)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _odds_to_dict(odds: GameOdds) -> dict:
    """Convert GameOdds to dict for JSON serialization."""
    return {
        "spread": odds.spread,
        "spread_odds": odds.spread_odds,
        "over_under": odds.over_under,
        "over_odds": odds.over_odds,
        "under_odds": odds.under_odds,
        "home_ml": odds.home_ml,
        "away_ml": odds.away_ml,
        "home_win_pct": odds.home_win_pct,
        "away_win_pct": odds.away_win_pct,
        "home_ats_record": odds.home_ats_record,
        "away_ats_record": odds.away_ats_record,
    }


def _play_to_dict(play: PlayEvent) -> dict:
    """Convert PlayEvent to dict for JSON serialization."""
    return {
        "event_id": play.event_id,
        "clock": play.clock,
        "period": play.period,
        "text": play.text,
        "team": play.team,
        "score_home": play.score_home,
        "score_away": play.score_away,
        "event_type": play.event_type,
    }


def _game_to_dict(game: GameInfo) -> dict:
    """Convert GameInfo to dict for JSON serialization."""
    return {
        "game_id": game.game_id,
        "home_team": game.home_team,
        "away_team": game.away_team,
        "home_abbr": game.home_abbr,
        "away_abbr": game.away_abbr,
        "home_score": game.home_score,
        "away_score": game.away_score,
        "status": game.status,
        "period": game.period,
        "clock": game.clock,
    }


def create_app() -> FastAPI:
    """Factory for embedding or testing."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=False)
