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

from src.analytics.backtester import BacktestResults, run_backtest, get_actual_game_results
from src.analytics.live_recommendations import build_live_recommendations
from src.analytics.prediction import predict_matchup
from src.analytics.stats_engine import get_scheduled_games, get_team_matchup_stats, TeamMatchupStats
from src.data.sync_service import (
    full_sync,
    sync_injuries,
    sync_injury_history,
    sync_live_scores,
    sync_schedule,
)
from src.data.nba_fetcher import get_current_season
from src.data.gamecast import (
    get_live_games,
    get_game_odds,
    get_box_score,
    get_play_by_play,
    GamecastClient,
    GameInfo,
    GameOdds,
    BoxScore,
    PlayEvent,
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
    from datetime import date as date_type
    
    lookup = _team_lookup()
    # Reverse lookup: abbr -> team_id
    abbr_to_id = {abbr: tid for tid, abbr in lookup.items()}
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
        
        # Sort by date ascending (today first)
        df = df.sort_values("game_date")
        
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
        })
    return result


@app.get("/matchups", response_class=HTMLResponse)
async def matchups(request: Request, home_team_id: int | None = None, away_team_id: int | None = None) -> HTMLResponse:
    teams = _team_list()
    games = await asyncio.to_thread(get_scheduled_games, 14)
    prediction = None
    home_stats = None
    away_stats = None
    backtest = None
    home_injury = None
    away_injury = None
    home_players = []
    away_players = []
    error: Optional[str] = None

    if home_team_id and away_team_id:
        try:
            # Get comprehensive team stats
            home_stats = await asyncio.to_thread(
                get_team_matchup_stats, home_team_id, away_team_id, True
            )
            away_stats = await asyncio.to_thread(
                get_team_matchup_stats, away_team_id, home_team_id, False
            )

            if home_stats.players and away_stats.players:
                # Calculate prediction
                home_proj = home_stats.projected_points + HOME_COURT_ADV
                away_proj = away_stats.projected_points
                prediction = {
                    "predicted_spread": home_proj - away_proj,
                    "predicted_total": home_proj + away_proj,
                    "home_projected": home_proj,
                    "away_projected": away_proj,
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
            "games": games,
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
    predictions: List[dict] = []
    avg_spread_err = 0.0
    avg_total_err = 0.0
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
            if results and results.predictions:
                predictions = _format_predictions(results)
                avg_spread_err = sum(abs(p.spread_error) for p in results.predictions) / len(results.predictions)
                avg_total_err = sum(abs(p.total_error) for p in results.predictions) / len(results.predictions)
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
            "predictions": predictions,
            "avg_spread_err": avg_spread_err,
            "avg_total_err": avg_total_err,
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


# --- Gamecast (ESPN real-time game data) ---

@app.get("/gamecast", response_class=HTMLResponse)
async def gamecast(request: Request, game_id: str | None = None) -> HTMLResponse:
    """Gamecast page with live play-by-play and odds."""
    games = await asyncio.to_thread(get_live_games)
    
    selected_game: GameInfo | None = None
    odds: GameOdds | None = None
    box: BoxScore | None = None
    recent_plays: List[PlayEvent] = []
    
    if game_id:
        # Find the selected game
        selected_game = next((g for g in games if g.game_id == game_id), None)
        
        # Fetch initial data
        odds = await asyncio.to_thread(get_game_odds, game_id)
        box = await asyncio.to_thread(get_box_score, game_id)
        recent_plays = await asyncio.to_thread(get_play_by_play, game_id, "")
        recent_plays = recent_plays[:20]  # Last 20 plays
    
    return templates.TemplateResponse(
        "gamecast.html",
        {
            "request": request,
            "games": games,
            "selected_game": selected_game,
            "game_id": game_id,
            "odds": odds,
            "box": box,
            "recent_plays": recent_plays,
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
