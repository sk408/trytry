"""FastAPI web application — 30+ routes, 18+ SSE endpoints."""

import asyncio
import json
import logging
import os
import threading
import time
from typing import Optional

from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import (
    HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.database import db
from src.database.migrations import init_db
from src.config import get_config

logger = logging.getLogger(__name__)

app = FastAPI(title="NBA Game Prediction System")

# Static files & templates
_static_dir = os.path.join(os.path.dirname(__file__), "static")
_template_dir = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(_static_dir, exist_ok=True)
os.makedirs(_template_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=_static_dir), name="static")
templates = Jinja2Templates(directory=_template_dir)

# Cancel events for long-running operations
_sync_cancel = threading.Event()
_optimize_cancel = threading.Event()
_pipeline_cancel = threading.Event()


# ---- Helper: SSE generator ----
def _sse_generator(fn, cancel_event=None):
    """Run fn in a thread, yield SSE events from its callback."""
    import queue
    q = queue.Queue()
    done = threading.Event()
    result = [None]

    def callback(msg):
        q.put(msg)

    def runner():
        try:
            result[0] = fn(callback=callback)
        except Exception as e:
            q.put(f"ERROR: {e}")
        finally:
            done.set()

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    async def generate():
        while not done.is_set() or not q.empty():
            try:
                msg = q.get(timeout=0.5)
                yield f"data: {msg}\n\n"
            except Exception:
                pass
            if cancel_event and cancel_event.is_set():
                yield "data: [CANCELLED]\n\n"
                break
        # Flush remaining
        while not q.empty():
            msg = q.get_nowait()
            yield f"data: {msg}\n\n"
        # Send result as JSON if available
        if result[0] is not None:
            try:
                yield f"data: [RESULTS_JSON]{json.dumps(result[0])}\n\n"
            except Exception:
                pass
        yield "data: [DONE]\n\n"

    return generate()


# ======== HTML PAGE ROUTES ========

@app.get("/", response_class=RedirectResponse)
async def index():
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    teams_count = db.fetch_one("SELECT COUNT(*) as c FROM teams")["c"]
    players_count = db.fetch_one("SELECT COUNT(*) as c FROM players")["c"]
    games_count = db.fetch_one("SELECT COUNT(*) as c FROM player_stats")["c"]
    try:
        injuries_count = db.fetch_one("SELECT COUNT(*) as c FROM injuries")["c"]
    except Exception:
        injuries_count = 0
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "teams": teams_count,
        "players": players_count,
        "game_logs": games_count,
        "injured": injuries_count,
    })


@app.post("/dashboard/sync")
async def dashboard_sync():
    return RedirectResponse(url="/dashboard", status_code=303)


@app.post("/dashboard/injuries")
async def dashboard_injuries():
    return RedirectResponse(url="/dashboard", status_code=303)


@app.post("/dashboard/injury-history")
async def dashboard_injury_history():
    return RedirectResponse(url="/dashboard", status_code=303)


@app.post("/dashboard/team-metrics")
async def dashboard_team_metrics():
    return RedirectResponse(url="/dashboard", status_code=303)


@app.post("/dashboard/player-impact")
async def dashboard_player_impact():
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/live", response_class=HTMLResponse)
async def live_page(request: Request):
    from src.data.live_scores import fetch_live_scores
    try:
        games = fetch_live_scores()
    except Exception:
        games = []
    return templates.TemplateResponse("live.html", {
        "request": request,
        "games": games,
    })


@app.get("/players", response_class=HTMLResponse)
async def players_page(request: Request):
    players = db.fetch_all("""
        SELECT p.*, t.abbreviation, i.status as injury_status, i.reason as injury_reason
        FROM players p
        LEFT JOIN teams t ON p.team_id = t.team_id
        LEFT JOIN injuries i ON p.player_id = i.player_id
        ORDER BY p.name
    """)
    injured = [dict(p) for p in players if p.get("injury_status")]
    manual = db.fetch_all("SELECT * FROM injuries WHERE source = 'manual'")
    return templates.TemplateResponse("players.html", {
        "request": request,
        "players": [dict(p) for p in players],
        "injured": injured,
        "manual_injuries": [dict(m) for m in manual],
    })


@app.post("/players/injury/add")
async def add_manual_injury(
    player_id: int = Form(...),
    player_name: str = Form(""),
    team_id: int = Form(0),
    status: str = Form("Out"),
    reason: str = Form(""),
):
    from src.data.injury_scraper import add_manual_injury
    add_manual_injury(player_id, player_name, team_id, status, reason)
    return RedirectResponse(url="/players", status_code=303)


@app.post("/players/injury/remove")
async def remove_manual_injury(player_id: int = Form(...)):
    from src.data.injury_scraper import remove_manual_injury
    remove_manual_injury(player_id)
    return RedirectResponse(url="/players", status_code=303)


@app.get("/schedule", response_class=HTMLResponse)
async def schedule_page(request: Request):
    from src.data.nba_fetcher import fetch_nba_cdn_schedule
    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
    try:
        schedule = fetch_nba_cdn_schedule()
        if schedule:
            schedule = [g for g in schedule
                       if today <= g.get("game_date", "") <= end]
    except Exception:
        schedule = []
    return templates.TemplateResponse("schedule.html", {
        "request": request,
        "schedule": schedule,
        "today": today,
    })


@app.get("/matchups", response_class=HTMLResponse)
async def matchups_page(request: Request,
                         home_team: Optional[int] = None,
                         away_team: Optional[int] = None):
    teams = db.fetch_all("SELECT team_id, abbreviation, name FROM teams ORDER BY abbreviation")
    prediction = None
    if home_team and away_team:
        try:
            from datetime import datetime
            from src.analytics.prediction import predict_matchup
            today = datetime.now().strftime("%Y-%m-%d")
            pred = predict_matchup(home_team, away_team, game_date=today)
            prediction = pred.__dict__
        except Exception as e:
            prediction = {"error": str(e)}
    return templates.TemplateResponse("matchups.html", {
        "request": request,
        "teams": [dict(t) for t in teams],
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
    })


@app.get("/accuracy", response_class=HTMLResponse)
async def accuracy_page(request: Request):
    from src.analytics.backtester import get_backtest_cache_age
    cache_age = get_backtest_cache_age()
    return templates.TemplateResponse("accuracy.html", {
        "request": request,
        "cache_age": cache_age,
    })


@app.get("/autotune", response_class=HTMLResponse)
async def autotune_page(request: Request):
    tuning = db.fetch_all("""
        SELECT tt.*, t.abbreviation
        FROM team_tuning tt
        JOIN teams t ON tt.team_id = t.team_id
        ORDER BY t.abbreviation
    """)
    return templates.TemplateResponse("autotune.html", {
        "request": request,
        "tuning": [dict(t) for t in tuning],
    })


@app.post("/autotune/clear")
async def clear_autotune():
    from src.analytics.autotune import clear_all_tuning
    clear_all_tuning()
    return RedirectResponse(url="/autotune", status_code=303)


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    from src.database.db import get_db_size
    config = get_config()
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "db_path": config.get("db_path", "data/nba.db"),
        "db_size": get_db_size(),
    })


@app.post("/admin/reset")
async def admin_reset():
    from src.database.db import delete_database
    delete_database()
    init_db()
    return RedirectResponse(url="/admin", status_code=303)


@app.get("/gamecast", response_class=HTMLResponse)
async def gamecast_page(request: Request, game_id: Optional[str] = None):
    return templates.TemplateResponse("gamecast.html", {
        "request": request,
        "game_id": game_id,
    })


# ======== SSE STREAMING ENDPOINTS ========

@app.get("/api/sync/data")
async def sse_sync_data():
    _sync_cancel.clear()
    from src.data.sync_service import full_sync
    return StreamingResponse(
        _sse_generator(full_sync, _sync_cancel),
        media_type="text/event-stream",
    )


@app.get("/api/sync/injuries")
async def sse_sync_injuries():
    from src.data.injury_scraper import sync_injuries
    return StreamingResponse(
        _sse_generator(sync_injuries),
        media_type="text/event-stream",
    )


@app.get("/api/sync/injury-history")
async def sse_sync_injury_history():
    from src.analytics.injury_history import infer_injuries_from_logs
    return StreamingResponse(
        _sse_generator(infer_injuries_from_logs),
        media_type="text/event-stream",
    )


@app.get("/api/sync/team-metrics")
async def sse_sync_team_metrics():
    from src.data.sync_service import sync_team_metrics
    return StreamingResponse(
        _sse_generator(sync_team_metrics),
        media_type="text/event-stream",
    )


@app.get("/api/sync/player-impact")
async def sse_sync_player_impact():
    from src.data.sync_service import sync_player_impact
    return StreamingResponse(
        _sse_generator(sync_player_impact),
        media_type="text/event-stream",
    )


@app.get("/api/sync/images")
async def sse_sync_images():
    from src.data.image_cache import preload_images
    return StreamingResponse(
        _sse_generator(preload_images),
        media_type="text/event-stream",
    )


@app.get("/api/backtest")
async def sse_backtest():
    from src.analytics.backtester import run_backtest
    return StreamingResponse(
        _sse_generator(run_backtest),
        media_type="text/event-stream",
    )


@app.get("/api/optimize")
async def sse_optimize():
    _optimize_cancel.clear()

    def _run(callback):
        from src.analytics.prediction import precompute_game_data
        from src.analytics.weight_optimizer import optimize_weights
        games = precompute_game_data(callback=callback)
        if not games:
            callback("No precomputed games available")
            return {"error": "no_data"}
        return optimize_weights(games, n_trials=200, callback=callback)

    return StreamingResponse(
        _sse_generator(_run, _optimize_cancel),
        media_type="text/event-stream",
    )


@app.get("/api/calibrate")
async def sse_calibrate():
    def _run(callback):
        from src.analytics.prediction import precompute_game_data
        from src.analytics.weight_optimizer import build_residual_calibration
        games = precompute_game_data(callback=callback)
        if not games:
            return {"error": "no_data"}
        return build_residual_calibration(games, callback=callback)
    return StreamingResponse(
        _sse_generator(_run),
        media_type="text/event-stream",
    )


@app.get("/api/feature-importance")
async def sse_feature_importance():
    def _run(callback):
        from src.analytics.prediction import precompute_game_data
        from src.analytics.weight_optimizer import compute_feature_importance
        games = precompute_game_data(callback=callback)
        if not games:
            return []
        return compute_feature_importance(games, callback=callback)
    return StreamingResponse(
        _sse_generator(_run),
        media_type="text/event-stream",
    )


@app.get("/api/grouped-feature-importance")
async def sse_grouped_feature_importance():
    def _run(callback):
        from src.analytics.prediction import precompute_game_data
        from src.analytics.weight_optimizer import compute_feature_importance
        games = precompute_game_data(callback=callback)
        if not games:
            return {}
        features = compute_feature_importance(games, callback=callback)
        # Group by category
        groups = {}
        for f in features:
            name = f["feature"]
            if "factor" in name or "rating" in name:
                cat = "Defense/Ratings"
            elif "ff_" in name or "four_factors" in name:
                cat = "Four Factors"
            elif "hustle" in name or "clutch" in name:
                cat = "Hustle/Clutch"
            elif "fatigue" in name or "pace" in name:
                cat = "Pace/Fatigue"
            else:
                cat = "Other"
            groups.setdefault(cat, []).append(f)
        return groups
    return StreamingResponse(
        _sse_generator(_run),
        media_type="text/event-stream",
    )


@app.get("/api/ml-feature-importance")
async def sse_ml_feature_importance():
    def _run(callback):
        from src.analytics.ml_model import get_shap_importance
        return get_shap_importance()
    return StreamingResponse(
        _sse_generator(_run),
        media_type="text/event-stream",
    )


@app.get("/api/fft-analysis")
async def sse_fft_analysis():
    def _run(callback):
        from src.analytics.backtester import run_backtest
        callback("Running backtest for FFT analysis...")
        results = run_backtest(callback=callback)
        per_game = results.get("per_game", [])
        if not per_game:
            return {"error": "no_data"}

        import numpy as np
        errors = [g["spread_error"] for g in per_game]
        if len(errors) < 10:
            return {"error": "insufficient_data"}

        fft = np.fft.fft(errors)
        freqs = np.fft.fftfreq(len(errors))
        power = np.abs(fft) ** 2

        # Top 10 frequency components
        top_idx = np.argsort(power[1:len(errors)//2])[::-1][:10] + 1
        components = []
        for idx in top_idx:
            components.append({
                "frequency": float(freqs[idx]),
                "period": round(1.0 / abs(freqs[idx]), 1) if freqs[idx] != 0 else 0,
                "power": float(power[idx]),
            })
        return {"components": components, "total_games": len(errors)}
    return StreamingResponse(
        _sse_generator(_run),
        media_type="text/event-stream",
    )


@app.get("/api/ml-train")
async def sse_ml_train():
    def _run(callback):
        from src.analytics.prediction import precompute_game_data
        from src.analytics.ml_model import train_models
        games = precompute_game_data(callback=callback)
        if not games:
            return {"error": "no_data"}
        return train_models(games, callback=callback)
    return StreamingResponse(
        _sse_generator(_run),
        media_type="text/event-stream",
    )


@app.get("/api/team-refinement")
async def sse_team_refinement():
    def _run(callback):
        from src.analytics.prediction import precompute_game_data
        from src.analytics.weight_optimizer import per_team_refinement
        games = precompute_game_data(callback=callback)
        if not games:
            return {"error": "no_data"}
        return per_team_refinement(games, n_trials=100, callback=callback)
    return StreamingResponse(
        _sse_generator(_run),
        media_type="text/event-stream",
    )


@app.get("/api/continuous-optimize")
async def sse_continuous_optimize():
    _optimize_cancel.clear()

    def _run(callback):
        from src.analytics.prediction import precompute_game_data
        from src.analytics.weight_optimizer import optimize_weights
        iteration = 0
        best_loss = float("inf")
        while not _optimize_cancel.is_set():
            iteration += 1
            callback(f"--- Iteration {iteration} ---")
            games = precompute_game_data(callback=callback)
            if not games:
                callback("No data, waiting...")
                time.sleep(30)
                continue
            result = optimize_weights(games, n_trials=200, callback=callback)
            loss = result.get("best_loss", float("inf"))
            if loss < best_loss:
                best_loss = loss
            callback(f"Iteration {iteration} complete: loss={loss:.3f} (best={best_loss:.3f})")
            if _optimize_cancel.is_set():
                break
            time.sleep(5)
        return {"iterations": iteration, "best_loss": best_loss}

    return StreamingResponse(
        _sse_generator(_run, _optimize_cancel),
        media_type="text/event-stream",
    )


@app.get("/api/optimize-all")
async def sse_optimize_all():
    _optimize_cancel.clear()

    def _run(callback):
        from src.analytics.prediction import precompute_game_data
        from src.analytics.weight_optimizer import optimize_weights, per_team_refinement
        callback("Phase 1: Global optimization...")
        games = precompute_game_data(callback=callback)
        if not games:
            return {"error": "no_data"}
        global_result = optimize_weights(games, n_trials=200, callback=callback)
        callback("Phase 2: Per-team refinement...")
        team_result = per_team_refinement(games, n_trials=100, callback=callback)
        return {"global": global_result, "team": team_result}

    return StreamingResponse(
        _sse_generator(_run, _optimize_cancel),
        media_type="text/event-stream",
    )


@app.get("/api/full-pipeline")
async def sse_full_pipeline():
    _pipeline_cancel.clear()
    from src.analytics.pipeline import run_full_pipeline
    return StreamingResponse(
        _sse_generator(run_full_pipeline, _pipeline_cancel),
        media_type="text/event-stream",
    )


@app.get("/api/gamecast/stream/{game_id}")
async def sse_gamecast_stream(game_id: str):
    async def generate():
        from src.data.gamecast import (
            fetch_espn_game_summary, get_espn_odds,
            get_espn_plays, get_espn_boxscore,
            normalize_espn_abbr,
        )
        while True:
            try:
                raw_summary = fetch_espn_game_summary(game_id)
                odds = get_espn_odds(game_id)
                plays = get_espn_plays(game_id)
                boxscore = get_espn_boxscore(game_id)

                # Parse summary into a simple dict for the frontend
                header = raw_summary.get("header", {})
                competitions = header.get("competitions", [{}])
                comp = competitions[0] if competitions else {}
                competitors = comp.get("competitors", [])
                home_c = next((c for c in competitors if c.get("homeAway") == "home"), {})
                away_c = next((c for c in competitors if c.get("homeAway") == "away"), {})
                summary = {
                    "home_team": normalize_espn_abbr(home_c.get("team", {}).get("abbreviation", "Home")),
                    "away_team": normalize_espn_abbr(away_c.get("team", {}).get("abbreviation", "Away")),
                    "home_score": int(home_c.get("score", 0) or 0),
                    "away_score": int(away_c.get("score", 0) or 0),
                    "status": comp.get("status", {}).get("type", {}).get("description", ""),
                }

                # Parse plays into simple list
                play_items = []
                if isinstance(plays, list):
                    for play in plays:
                        items = play.get("items", [play])
                        for item in items:
                            if isinstance(item, dict):
                                text = item.get("text", "")
                                clock = item.get("clock", {})
                                clock_str = clock.get("displayValue", "") if isinstance(clock, dict) else str(clock)
                                play_items.append({"text": text, "clock": clock_str})

                data = {
                    "summary": summary,
                    "odds": odds,
                    "plays": play_items[-10:] if play_items else [],
                    "boxscore": boxscore,
                }
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(10)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ======== REST API ENDPOINTS ========

@app.post("/api/sync/cancel")
async def cancel_sync():
    _sync_cancel.set()
    return JSONResponse({"status": "cancel_requested"})


@app.post("/api/continuous-optimize/cancel")
async def cancel_continuous_optimize():
    _optimize_cancel.set()
    return JSONResponse({"status": "cancel_requested"})


@app.post("/api/pipeline/cancel")
async def cancel_pipeline():
    _pipeline_cancel.set()
    from src.analytics.pipeline import request_cancel
    request_cancel()
    return JSONResponse({"status": "ok"})


@app.post("/api/weights/clear")
async def clear_weights():
    from src.analytics.weight_config import clear_all_weights
    clear_all_weights()
    return JSONResponse({"status": "cleared"})


@app.get("/api/weights")
async def get_weights():
    from src.analytics.weight_config import get_weight_config
    w = get_weight_config()
    return JSONResponse(w.to_dict())


@app.get("/api/calibration")
async def get_calibration():
    spread = db.fetch_all("SELECT * FROM residual_calibration ORDER BY bin_low")
    total = db.fetch_all("SELECT * FROM residual_calibration_total ORDER BY bin_low")
    return JSONResponse({
        "spread": [dict(r) for r in spread] if spread else [],
        "total": [dict(r) for r in total] if total else [],
    })


@app.get("/api/backtest-cache-age")
async def backtest_cache_age():
    from src.analytics.backtester import get_backtest_cache_age
    age = get_backtest_cache_age()
    return JSONResponse({"age_minutes": round(age, 1) if age is not None else None})


@app.get("/api/gamecast/games")
async def gamecast_games():
    from src.data.gamecast import fetch_espn_scoreboard
    try:
        games = fetch_espn_scoreboard()
        return JSONResponse(games)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/gamecast/odds/{game_id}")
async def gamecast_odds(game_id: str):
    from src.data.gamecast import get_espn_odds
    try:
        odds = get_espn_odds(game_id)
        return JSONResponse(odds)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/gamecast/boxscore/{game_id}")
async def gamecast_boxscore(game_id: str):
    from src.data.gamecast import get_espn_boxscore
    try:
        box = get_espn_boxscore(game_id)
        return JSONResponse(box)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ======== Startup ========

@app.on_event("startup")
async def startup():
    init_db()
    logger.info("NBA Prediction System started")
