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
_data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_logos_dir = os.path.join(_data_dir, "cache", "team_logos")
_photos_dir = os.path.join(_data_dir, "cache", "player_photos")
os.makedirs(_static_dir, exist_ok=True)
os.makedirs(_template_dir, exist_ok=True)
os.makedirs(_logos_dir, exist_ok=True)
os.makedirs(_photos_dir, exist_ok=True)

app.mount("/images/logos", StaticFiles(directory=_logos_dir), name="logos")
app.mount("/images/players", StaticFiles(directory=_photos_dir), name="player_photos")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")
templates = Jinja2Templates(directory=_template_dir)

# Cancel events for long-running operations
_sync_cancel = threading.Event()
_optimize_cancel = threading.Event()
_pipeline_cancel = threading.Event()

# Locks to prevent concurrent SSE operations on the same endpoint
_sync_lock = threading.Lock()
_optimize_lock = threading.Lock()
_pipeline_lock = threading.Lock()


# ---- Helper: SSE generator ----
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def _sse_generator(fn, cancel_event=None, lock=None):
    """Run fn in a thread, yield SSE events from its callback.

    If *lock* is provided it is released when the stream finishes (the caller
    must have already acquired it before calling this function).
    """
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
        try:
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
                    yield f"data: [RESULTS_JSON]{json.dumps(result[0], cls=NumpyEncoder)}\n\n"
                except Exception:
                    pass
            yield "data: [DONE]\n\n"
        finally:
            if lock is not None:
                lock.release()

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


@app.get("/players", response_class=HTMLResponse)
async def players_page(request: Request, page: int = Query(1, ge=1)):
    import re
    from datetime import datetime

    def _days_until_return(expected_return: str):
        if not expected_return or len(expected_return) < 4:
            return None
        try:
            now = datetime.now()
            m = re.match(r"([A-Za-z]+)\s+(\d+)", expected_return)
            if m:
                month_str, day_str = m.group(1), m.group(2)
                for fmt in ("%b %d %Y", "%B %d %Y"):
                    for year in (now.year, now.year + 1):
                        try:
                            dt = datetime.strptime(f"{month_str} {day_str} {year}", fmt)
                            delta = (dt - now).days
                            if delta >= -30:
                                return max(0, delta)
                        except ValueError:
                            continue
        except Exception:
            pass
        return None

    def _estimate_play_probability(status, expected_return, detail):
        low_status = (status or "").lower()
        low_detail = (detail or "").lower()
        season_ending = any(kw in low_detail for kw in (
            "season-ending", "remainder of the", "rest of the 2025",
            "rest of the 2026", "torn acl", "torn achilles",
            "indefinitely", "suspended",
        ))
        if expected_return and expected_return.strip().lower().startswith("oct"):
            season_ending = True
        if season_ending:
            return (0.0, "OUT FOR SEASON", "danger")
        if "surgery" in low_detail and "successful" not in low_detail:
            return (0.0, "OUT (surgery)", "danger")
        return_days = _days_until_return(expected_return or "")
        if low_status in ("out", "o"):
            if return_days is not None:
                if return_days <= 0:
                    return (0.15, "OUT (back soon)", "warning")
                elif return_days <= 3:
                    return (0.10, f"OUT ~{return_days}d", "danger")
                elif return_days <= 14:
                    return (0.02, f"OUT ~{return_days}d", "danger")
                else:
                    wks = return_days // 7
                    return (0.0, f"OUT ~{wks}wk", "danger")
            return (0.0, "OUT", "danger")
        if low_status == "doubtful":
            return (0.15, "DOUBTFUL 15%", "danger")
        if low_status in ("questionable", "gtd"):
            return (0.50, "QUESTIONABLE 50%", "warning")
        if low_status == "day-to-day":
            if return_days is not None and return_days <= 1:
                return (0.65, "DAY-TO-DAY 65%", "warning")
            return (0.55, "DAY-TO-DAY 55%", "warning")
        if low_status == "probable":
            return (0.85, "PROBABLE 85%", "success")
        return (0.25, (status or "").upper(), "muted")

    players_raw = db.fetch_all("""
        SELECT p.*, t.abbreviation,
               i.status as injury_status, i.reason as injury_reason,
               i.expected_return as injury_return
        FROM players p
        LEFT JOIN teams t ON p.team_id = t.team_id
        LEFT JOIN injuries i ON p.player_id = i.player_id
        ORDER BY p.name
    """)
    all_players = []
    injured = []
    for p in players_raw:
        pdict = dict(p)
        mpg = pdict.get("mpg", 0) or 0
        ppg = pdict.get("ppg", 0) or 0
        # Impact classification
        if mpg >= 25:
            pdict["impact"] = "KEY"
            pdict["impact_color"] = "danger"
        elif mpg >= 15:
            pdict["impact"] = "ROTATION"
            pdict["impact_color"] = "warning"
        else:
            pdict["impact"] = "BENCH"
            pdict["impact_color"] = "muted"
        # Play probability for injured
        if pdict.get("injury_status"):
            prob, label, color = _estimate_play_probability(
                pdict.get("injury_status", ""),
                pdict.get("injury_return", ""),
                pdict.get("injury_reason", ""),
            )
            pdict["play_probability"] = prob
            pdict["availability_label"] = label
            pdict["availability_color"] = color
            injured.append(pdict)
        all_players.append(pdict)

    # Sort injured by MPG descending (highest impact first)
    injured.sort(key=lambda x: x.get("mpg", 0) or 0, reverse=True)

    # Pagination
    per_page = 100
    total_players = len(all_players)
    total_pages = max(1, (total_players + per_page - 1) // per_page)
    page = min(page, total_pages)
    start = (page - 1) * per_page
    paginated = all_players[start:start + per_page]

    manual = db.fetch_all("SELECT * FROM injuries WHERE source = 'manual'")
    return templates.TemplateResponse("players.html", {
        "request": request,
        "players": paginated,
        "injured": injured,
        "manual_injuries": [dict(m) for m in manual],
        "page": page,
        "total_pages": total_pages,
        "total_players": total_players,
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
async def schedule_page(request: Request,
                        start_date: Optional[str] = None,
                        days: int = Query(14, ge=1, le=60)):
    from src.data.nba_fetcher import fetch_nba_cdn_schedule
    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    start = start_date if start_date else today
    try:
        end_dt = datetime.strptime(start, "%Y-%m-%d") + timedelta(days=days)
        end = end_dt.strftime("%Y-%m-%d")
    except Exception:
        end = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        schedule = fetch_nba_cdn_schedule()
        if schedule:
            schedule = [g for g in schedule
                       if start <= g.get("game_date", "") <= end]
    except Exception:
        schedule = []
    return templates.TemplateResponse("schedule.html", {
        "request": request,
        "schedule": schedule,
        "today": today,
        "start_date": start,
        "days": days,
    })


@app.get("/matchups", response_class=HTMLResponse)
async def matchups_page(request: Request,
                         home_team: Optional[int] = None,
                         away_team: Optional[int] = None):
    teams = db.fetch_all("SELECT team_id, abbreviation, name FROM teams ORDER BY abbreviation")
    prediction = None
    rosters = {"home": [], "away": []}

    # Load upcoming games for game picker
    upcoming_games = []
    try:
        from src.data.nba_fetcher import fetch_nba_cdn_schedule
        from datetime import datetime, timedelta
        today = datetime.now().strftime("%Y-%m-%d")
        end = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        sched = fetch_nba_cdn_schedule()
        if sched:
            upcoming_games = [g for g in sched
                             if today <= g.get("game_date", "") <= end]
    except Exception:
        pass

    if home_team and away_team:
        try:
            from datetime import datetime
            from src.analytics.prediction import predict_matchup
            today = datetime.now().strftime("%Y-%m-%d")
            pred = predict_matchup(home_team, away_team, game_date=today)
            prediction = pred.__dict__
            # Add formatted breakdown for display
            prediction["breakdown"] = {
                "Home Court Advantage": pred.home_court_advantage,
                "Fatigue Adjustment": pred.fatigue_adj,
                "Turnover Differential": pred.turnover_adj,
                "Rebound Differential": pred.rebound_adj,
                "Rating Matchup": pred.rating_matchup_adj,
                "Four Factors": pred.four_factors_adj,
                "Clutch Factor": pred.clutch_adj,
                "Hustle (Spread)": pred.hustle_adj,
                "Pace (Total)": pred.pace_adj,
                "Def. Disruption (Total)": pred.defensive_disruption,
                "OREB Boost (Total)": pred.oreb_boost,
                "Hustle (Total)": pred.hustle_total_adj,
                "Sharp Money": pred.adjustments.get("sharp_money", 0),
                "ESPN Blend": pred.espn_blend_adj,
                "ML Ensemble": pred.ml_blend_adj,
            }
            # Sharp money raw data for display
            if pred.sharp_home_public > 0:
                prediction["sharp_money_info"] = {
                    "spread_public": f"H {pred.sharp_home_public}% / A {100 - pred.sharp_home_public}%",
                    "spread_money": f"H {pred.sharp_home_money}% / A {100 - pred.sharp_home_money}%",
                    "edge": pred.sharp_home_money - pred.sharp_home_public,
                }
            # Compute win probability from spread
            import math
            spread = pred.predicted_spread or 0
            prediction["home_win_prob"] = 100.0 / (1.0 + math.exp(-0.15 * spread)) if spread else 50.0
            prediction["spread"] = pred.predicted_spread
            prediction["total"] = pred.predicted_total
            prediction["home_score"] = pred.predicted_home_score
            prediction["away_score"] = pred.predicted_away_score
        except Exception as e:
            prediction = {"error": str(e)}

        # Load rosters with injury info and stats
        for side, tid in [("home", home_team), ("away", away_team)]:
            try:
                players = db.fetch_all("""
                    SELECT p.player_id, p.name as player_name, p.position,
                           t.abbreviation,
                           i.status as injury_status, i.reason as injury_reason,
                           COALESCE((SELECT AVG(points) FROM (
                                     SELECT points FROM player_stats ps
                                     WHERE ps.player_id = p.player_id
                                     ORDER BY ps.game_date DESC LIMIT 15)), 0) as ppg,
                           COALESCE((SELECT AVG(minutes) FROM (
                                     SELECT minutes FROM player_stats ps
                                     WHERE ps.player_id = p.player_id
                                     ORDER BY ps.game_date DESC LIMIT 10)), 0) as mpg
                    FROM players p
                    LEFT JOIN teams t ON p.team_id = t.team_id
                    LEFT JOIN injuries i ON p.player_id = i.player_id
                    WHERE p.team_id = ?
                    ORDER BY COALESCE((SELECT AVG(minutes) FROM (
                                       SELECT minutes FROM player_stats ps
                                       WHERE ps.player_id = p.player_id
                                       ORDER BY ps.game_date DESC LIMIT 10)), 0) DESC
                """, (tid,))
                roster_list = []
                for pl in players:
                    pdict = dict(pl)
                    mpg = pdict.get("mpg", 0) or 0
                    ppg = pdict.get("ppg", 0) or 0
                    # Impact classification
                    if mpg >= 25:
                        pdict["impact"] = "KEY"
                        pdict["impact_color"] = "danger"
                    elif mpg >= 15:
                        pdict["impact"] = "ROTATION"
                        pdict["impact_color"] = "warning"
                    else:
                        pdict["impact"] = "BENCH"
                        pdict["impact_color"] = "muted"
                    # Injury impact estimate
                    pdict["injury_impact"] = ""
                    if pdict.get("injury_status"):
                        status_lower = (pdict["injury_status"] or "").lower()
                        if status_lower in ("out", "o"):
                            pdict["injury_impact"] = f"-{ppg:.1f} pts"
                        elif status_lower == "doubtful":
                            pdict["injury_impact"] = f"~-{ppg*0.8:.1f} pts"
                        elif status_lower in ("questionable", "gtd", "day-to-day"):
                            pdict["injury_impact"] = f"~-{ppg*0.4:.1f} pts"
                        elif status_lower == "probable":
                            pdict["injury_impact"] = "Likely plays"
                    roster_list.append(pdict)
                rosters[side] = roster_list
            except Exception:
                pass

    return templates.TemplateResponse("matchups.html", {
        "request": request,
        "teams": [dict(t) for t in teams],
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
        "rosters": rosters,
        "upcoming_games": upcoming_games,
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
        "sync_freshness_hours": config.get("sync_freshness_hours", 4),
        "optimizer_log_interval": config.get("optimizer_log_interval", 300),
    })


@app.post("/admin/settings")
async def admin_save_settings(request: Request):
    from src.config import set_value
    data = await request.json()
    if "sync_freshness_hours" in data:
        set_value("sync_freshness_hours", max(1, min(168, int(data["sync_freshness_hours"]))))
    if "optimizer_log_interval" in data:
        set_value("optimizer_log_interval", max(1, min(3000, int(data["optimizer_log_interval"]))))
    return {"status": "ok"}


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
async def sse_sync_data(force: bool = False):
    if not _sync_lock.acquire(blocking=False):
        return JSONResponse({"error": "Sync already running"}, status_code=409)
    _sync_cancel.clear()
    from src.data.sync_service import full_sync
    from functools import partial
    fn = partial(full_sync, force=force) if force else full_sync
    return StreamingResponse(
        _sse_generator(fn, _sync_cancel, lock=_sync_lock),
        media_type="text/event-stream",
    )


@app.get("/api/odds/sync")
async def sse_odds_sync():
    from src.data.odds_sync import backfill_odds
    _sync_cancel.clear()
    return StreamingResponse(
        _sse_generator(backfill_odds, _sync_cancel),
        media_type="text/event-stream",
    )


@app.get("/api/odds/status")
async def odds_status():
    from src.database import db
    total_dates = db.fetch_one("SELECT COUNT(DISTINCT game_date) as c FROM player_stats")
    odds_dates = db.fetch_one("SELECT COUNT(DISTINCT game_date) as c FROM game_odds")
    return JSONResponse({
        "dates_with_games": total_dates["c"] if total_dates else 0,
        "dates_with_odds": odds_dates["c"] if odds_dates else 0
    })


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


@app.get("/api/sync/nuke-resync")
async def sse_nuke_resync():
    """Nuke all synced data then full force resync."""
    _sync_cancel.clear()

    def _nuke_and_resync(callback=None):
        from src.data.sync_service import nuke_synced_data, full_sync
        nuke_synced_data(callback=callback)
        if callback:
            callback("\nStarting full force resync from scratch...")
        full_sync(callback=callback, force=True)

    return StreamingResponse(
        _sse_generator(_nuke_and_resync, _sync_cancel),
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
async def sse_optimize(continuous: bool = False, target: str = "ats"):
    if not _optimize_lock.acquire(blocking=False):
        return JSONResponse({"error": "Optimize already running"}, status_code=409)
    _optimize_cancel.clear()

    def _run(callback):
        from src.analytics.prediction import precompute_game_data
        from src.analytics.weight_optimizer import optimize_weights
        from src.analytics.weight_config import get_weight_config, save_weight_config, save_snapshot
        import itertools
        
        games = precompute_game_data(callback=callback)
        if not games:
            callback("No precomputed games available")
            return {"error": "no_data"}
        
        if target == "all":
            targets = ["ats", "roi", "ml"]
        else:
            targets = [target]

        if continuous:
            target_iter = itertools.cycle(targets)
            n_trials = 2000 if target == "all" else 1000000 
        else:
            target_iter = iter(targets)
            n_trials = 2000

        saved_weights = {}
        last_result = None

        for t in target_iter:
            if _optimize_cancel.is_set():
                break
                
            if t in saved_weights:
                save_weight_config(saved_weights[t])
                
            if target == "all":
                callback(f"=== Optimizing target: {t.upper()} ===")
            callback(f"Optimizing weights ({n_trials} trials, target={t})...")
            
            last_result = optimize_weights(
                games, 
                n_trials=n_trials, 
                callback=callback, 
                is_cancelled=lambda: _optimize_cancel.is_set(),
                target=t
            )
            
            saved_weights[t] = get_weight_config()
            
            if last_result.get("improved", False):
                snap_name = f"Best_{t.upper()}_Weights"
                metrics = {
                    "ats_rate": last_result.get("ats_rate", 0),
                    "ats_roi": last_result.get("ats_roi", 0),
                    "ml_roi": last_result.get("ml_roi", 0),
                    "loss": last_result.get("loss", 0),
                }
                save_snapshot(snap_name, notes=f"Auto-saved by optimizer ({t.upper()})", metrics=metrics)
                
        return last_result

    return StreamingResponse(
        _sse_generator(_run, _optimize_cancel, lock=_optimize_lock),
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
        errors = [g.get("spread_error", 0) for g in per_game]
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
    if not _optimize_lock.acquire(blocking=False):
        return JSONResponse({"error": "Optimize already running"}, status_code=409)
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
        _sse_generator(_run, _optimize_cancel, lock=_optimize_lock),
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
    if not _pipeline_lock.acquire(blocking=False):
        return JSONResponse({"error": "Pipeline already running"}, status_code=409)
    _pipeline_cancel.clear()
    from src.analytics.pipeline import run_full_pipeline
    return StreamingResponse(
        _sse_generator(run_full_pipeline, _pipeline_cancel, lock=_pipeline_lock),
        media_type="text/event-stream",
    )


@app.get("/api/retune")
async def sse_retune():
    if not _pipeline_lock.acquire(blocking=False):
        return JSONResponse({"error": "Pipeline already running"}, status_code=409)
    _pipeline_cancel.clear()
    from src.analytics.pipeline import run_retune
    return StreamingResponse(
        _sse_generator(run_retune, _pipeline_cancel, lock=_pipeline_lock),
        media_type="text/event-stream",
    )


@app.get("/api/gamecast/stream/{game_id}")
async def sse_gamecast_stream(game_id: str):
    async def generate():
        from src.data.gamecast import (
            fetch_espn_game_summary, get_espn_odds,
            get_espn_plays, get_espn_boxscore,
            normalize_espn_abbr,
            FastcastWebSocket,
        )
        # Start Fastcast WebSocket for real-time change detection
        ws_event = threading.Event()
        ws = None
        try:
            ws = FastcastWebSocket(
                game_id,
                on_data_changed=lambda: ws_event.set(),
            )
            ws.start()
        except Exception:
            ws = None

        # Resolve NBA team IDs once
        _team_cache = {}
        _player_cache = {}

        def _resolve_tid(abbr):
            nba = normalize_espn_abbr(abbr)
            if nba in _team_cache:
                return _team_cache[nba]
            try:
                row = db.fetch_one(
                    "SELECT team_id FROM teams WHERE abbreviation = ?", (nba,))
                tid = row["team_id"] if row else None
            except Exception:
                tid = None
            _team_cache[nba] = tid
            return tid

        def _resolve_pid(name):
            if not name:
                return None
            if name in _player_cache:
                return _player_cache[name]
            try:
                row = db.fetch_one("SELECT player_id FROM players WHERE name = ?", (name,))
                pid = row["player_id"] if row else None
            except Exception:
                pid = None
            _player_cache[name] = pid
            return pid

        status_state = "in"  # assume live until proven otherwise
        while True:
            try:
                raw_summary = fetch_espn_game_summary(game_id)
                plays = get_espn_plays(game_id)
                boxscore = get_espn_boxscore(game_id)

                # Parse header
                header = raw_summary.get("header", {})
                competitions = header.get("competitions", [{}])
                comp = competitions[0] if competitions else {}
                competitors = comp.get("competitors", [])
                home_c = next((c for c in competitors if c.get("homeAway") == "home"), {})
                away_c = next((c for c in competitors if c.get("homeAway") == "away"), {})

                home_abbr = normalize_espn_abbr(home_c.get("team", {}).get("abbreviation", "HOME"))
                away_abbr = normalize_espn_abbr(away_c.get("team", {}).get("abbreviation", "AWAY"))
                
                # Fetch odds (try ActionNetwork for live, fallback to ESPN)
                try:
                    from src.data.gamecast import get_actionnetwork_odds
                    odds = get_actionnetwork_odds(home_abbr, away_abbr)
                    if not odds or not odds.get("spread"):
                        odds = get_espn_odds(game_id)
                except Exception:
                    odds = get_espn_odds(game_id)
                home_score = int(home_c.get("score", 0) or 0)
                away_score = int(away_c.get("score", 0) or 0)
                home_tid = _resolve_tid(home_abbr)
                away_tid = _resolve_tid(away_abbr)
                home_espn_id = str(home_c.get("team", {}).get("id", ""))
                away_espn_id = str(away_c.get("team", {}).get("id", ""))

                home_timeouts = home_c.get("timeoutsRemaining", -1)
                away_timeouts = away_c.get("timeoutsRemaining", -1)
                home_bonus = home_c.get("linescores", [{}])[-1].get("isBonus", False) if home_c.get("linescores") else False
                away_bonus = away_c.get("linescores", [{}])[-1].get("isBonus", False) if away_c.get("linescores") else False

                status_detail = comp.get("status", {})
                status_text = status_detail.get("type", {}).get("description", "")
                status_state = status_detail.get("type", {}).get("state", "")
                period = int(status_detail.get("period", 0) or 0)
                clock_str = status_detail.get("displayClock", "0:00")

                # Quarter scores
                home_quarters = []
                away_quarters = []
                for ls_comp in competitors:
                    linescores = ls_comp.get("linescores", [])
                    quarters = [int(q.get("displayValue", 0) or 0) for q in linescores]
                    if ls_comp.get("homeAway") == "home":
                        home_quarters = quarters
                    else:
                        away_quarters = quarters

                summary = {
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_team_id": home_tid,
                    "away_team_id": away_tid,
                    "home_espn_id": home_espn_id,
                    "away_espn_id": away_espn_id,
                    "status": status_text,
                    "status_state": status_state,
                    "clock": clock_str,
                    "period": period,
                    "home_quarters": home_quarters,
                    "away_quarters": away_quarters,
                    "home_timeouts": home_timeouts,
                    "away_timeouts": away_timeouts,
                    "home_bonus": home_bonus,
                    "away_bonus": away_bonus,
                }

                # Model prediction
                prediction = None
                if home_tid and away_tid:
                    try:
                        from src.analytics.live_prediction import live_predict
                        import math
                        q = 0
                        mins_el = 0.0
                        if status_state == "in":
                            q = min(max(0, period - 1), 4)
                            try:
                                clock_clean = clock_str.replace(" ", "")
                                if ":" in clock_clean:
                                    parts = clock_clean.split(":")
                                    mins_left = float(parts[0])
                                    secs_left = float(parts[1])
                                else:
                                    try:
                                        mins_left = 0
                                        secs_left = float(clock_clean)
                                    except ValueError:
                                        mins_left = 12 if period <= 4 else 5
                                        secs_left = 0
                                period_len = 12 * 60 if period <= 4 else 5 * 60
                                mins_el = ((period - 1) * 12.0 if period <= 4
                                           else 48.0 + (period - 5) * 5.0)
                                elapsed_in_q = period_len - (mins_left * 60 + secs_left)
                                mins_el += max(0, elapsed_in_q) / 60.0
                            except Exception:
                                mins_el = min(q, 4) * 12.0
                        elif status_state == "post":
                            q = 4
                            ot = max(0, period - 4) if period > 4 else 0
                            mins_el = 48.0 + ot * 5.0

                        pred = live_predict(
                            home_team_id=home_tid, away_team_id=away_tid,
                            home_score=home_score, away_score=away_score,
                            quarter=q, minutes_elapsed=mins_el,
                        )
                        spread = pred.get("spread", 0)
                        home_wp = 100.0 / (1.0 + math.exp(-0.15 * spread)) if spread else 50.0
                        if status_state == "post":
                            home_wp = 100.0 if home_score > away_score else (0.0 if away_score > home_score else 50.0)
                        prediction = {
                            "spread": round(spread, 1) if spread else 0,
                            "total": round(pred.get("total", 0), 1),
                            "home_score": round(pred.get("home_projected", 0), 1),
                            "away_score": round(pred.get("away_projected", 0), 1),
                            "home_win_prob": round(home_wp, 1),
                        }
                    except Exception:
                        pass

                # Parse plays with team attribution
                play_items = []
                if isinstance(plays, list):
                    for play in plays:
                        items = play.get("items", [play])
                        for item in items:
                            if isinstance(item, dict) and item.get("text"):
                                clock = item.get("clock", {})
                                clock_val = clock.get("displayValue", "") if isinstance(clock, dict) else str(clock)
                                per = item.get("period", {})
                                per_num = per.get("number", 0) if isinstance(per, dict) else 0
                                espn_tid = str(item.get("team", {}).get("id", ""))
                                team_id = None
                                if espn_tid == home_espn_id:
                                    team_id = home_tid
                                elif espn_tid == away_espn_id:
                                    team_id = away_tid
                                play_items.append({
                                    "text": item.get("text", ""),
                                    "clock": clock_val,
                                    "period": per_num,
                                    "scoring": item.get("scoringPlay", False),
                                    "home_score": item.get("homeScore", ""),
                                    "away_score": item.get("awayScore", ""),
                                    "team_id": team_id,
                                    "coordinate": item.get("coordinate"),
                                    "shootingPlay": item.get("shootingPlay", False),
                                })
                                
                # Calculate drives scored and scoring runs
                home_drives = 0
                away_drives = 0
                home_drives_scored = 0
                away_drives_scored = 0
                current_run_team = None
                current_run_pts = 0
                
                # play_items are usually chronological, but we reverse them just in case they aren't?
                # Actually, ESPN summary returns plays chronological.
                for p in play_items:
                    team = p.get("team_id")
                    text = p.get("text", "").lower()
                    scoring = p.get("scoring", False)
                    
                    if "driving" in text:
                        if team == home_tid:
                            home_drives += 1
                            if scoring:
                                home_drives_scored += 1
                        elif team == away_tid:
                            away_drives += 1
                            if scoring:
                                away_drives_scored += 1
                                
                    if not scoring:
                        continue
                        
                    # Calculate points from this play
                    pts = 0
                    if "free throw" in text:
                        pts = 1
                    elif "3-pt" in text or "three point" in text:
                        pts = 3
                    else:
                        pts = 2
                        
                    # Calculate unanswered run
                    if team == current_run_team:
                        current_run_pts += pts
                    else:
                        current_run_team = team
                        current_run_pts = pts
                        
                run_text = f"{current_run_pts}-0 run" if current_run_pts >= 5 else ""
                if current_run_pts >= 5 and current_run_team:
                    run_team_abbr = home_abbr if current_run_team == home_tid else away_abbr
                    run_text = f"{run_team_abbr} on a {run_text}"
                else:
                    run_text = ""
                    
                # Calculate Game Possessions from Team Box Score
                box_teams = boxscore.get("teams", []) if isinstance(boxscore, dict) else []
                home_poss = 0
                away_poss = 0
                home_poss_scored = 0
                away_poss_scored = 0
                for tb in box_teams:
                    tb_tid = str(tb.get("team", {}).get("id", ""))
                    side = "home" if tb_tid == home_espn_id else "away"
                    stats_blocks = tb.get("statistics", [])
                    if not stats_blocks:
                        continue
                    stats_dict = {}
                    for s in stats_blocks:
                        if "abbreviation" in s:
                            stats_dict[s["abbreviation"]] = s.get("displayValue")
                        elif "label" in s:
                            stats_dict[s["label"]] = s.get("displayValue")
                    
                    try:
                        fg = stats_dict.get("FG", "0-0").split("-")
                        fgm = int(fg[0]) if len(fg) > 0 and fg[0].isdigit() else 0
                        fga = int(fg[1]) if len(fg) > 1 and fg[1].isdigit() else 0
                        
                        ft = stats_dict.get("FT", "0-0").split("-")
                        ftm = int(ft[0]) if len(ft) > 0 and ft[0].isdigit() else 0
                        fta = int(ft[1]) if len(ft) > 1 and ft[1].isdigit() else 0
                        
                        orb = int(stats_dict.get("OR", stats_dict.get("Offensive Rebounds", 0)))
                        tov = int(stats_dict.get("TO", stats_dict.get("Turnovers", 0)))
                        
                        poss = round(fga + 0.44 * fta - orb + tov)
                        scored = round(fgm + (ftm / 2))  # Proxy for unique scoring possessions
                        
                        if side == "home":
                            home_poss = poss
                            home_poss_scored = scored
                        else:
                            away_poss = poss
                            away_poss_scored = scored
                    except Exception:
                        pass
                
                summary.update({
                    "home_drives": home_drives,
                    "away_drives": away_drives,
                    "home_drives_scored": home_drives_scored,
                    "away_drives_scored": away_drives_scored,
                    "home_poss": home_poss,
                    "away_poss": away_poss,
                    "home_poss_scored": home_poss_scored,
                    "away_poss_scored": away_poss_scored,
                    "current_run": run_text,
                })

                # Parse box score into simple format
                box_parsed = {"home": [], "away": []}
                box_players = boxscore.get("players", []) if isinstance(boxscore, dict) else []
                for tb in box_players:
                    tb_tid = str(tb.get("team", {}).get("id", ""))
                    side = "home" if tb_tid == home_espn_id else "away"
                    stats_blocks = tb.get("statistics", [])
                    if not stats_blocks:
                        continue
                    labels = stats_blocks[0].get("labels", [])
                    for ath in stats_blocks[0].get("athletes", []):
                        ainfo = ath.get("athlete", {})
                        stats = ath.get("stats", [])
                        smap = dict(zip(labels, stats)) if len(labels) == len(stats) else {}
                        
                        name = ainfo.get("displayName", "")
                        pid = _resolve_pid(name)
                        headshot = f"/images/players/{pid}.png" if pid else ainfo.get("headshot", {}).get("href", "")
                        
                        box_parsed[side].append({
                            "name": name,
                            "id": ainfo.get("id", ""),
                            "headshot": headshot,
                            "active": ath.get("active", False),
                            "min": smap.get("MIN", ""),
                            "pts": smap.get("PTS", ""),
                            "reb": smap.get("REB", ""),
                            "ast": smap.get("AST", ""),
                            "stl": smap.get("STL", ""),
                            "blk": smap.get("BLK", ""),
                            "fg": smap.get("FG", ""),
                            "threept": smap.get("3PT", ""),
                            "plusminus": smap.get("+/-", ""),
                            "pf": smap.get("PF", ""),
                        })

                # Calculate total team fouls from player box scores
                home_fouls = 0
                away_fouls = 0
                for side, p_list in [("home", box_parsed["home"]), ("away", box_parsed["away"])]:
                    for p in p_list:
                        try:
                            f = int(p["pf"]) if p["pf"] else 0
                            if side == "home":
                                home_fouls += f
                            else:
                                away_fouls += f
                        except ValueError:
                            pass

                summary.update({
                    "home_fouls": home_fouls,
                    "away_fouls": away_fouls,
                })

                data = {
                    "summary": summary,
                    "odds": odds,
                    "prediction": prediction,
                    "plays": play_items[-50:] if play_items else [],
                    "boxscore": box_parsed,
                }
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            # Stop streaming for finished games after sending final data
            if status_state == "post":
                if ws:
                    ws.stop()
                return

            # Wait for WebSocket change notification or fallback timeout
            ws_event.clear()
            fallback_sec = 10 if status_state == "in" else 30
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: ws_event.wait(fallback_sec)
                )
            except (GeneratorExit, asyncio.CancelledError):
                # Client disconnected — clean up WebSocket
                if ws:
                    ws.stop()
                return
            except Exception:
                await asyncio.sleep(fallback_sec)

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
    try:
        spread = db.fetch_all("SELECT * FROM residual_calibration ORDER BY bin_low")
    except Exception:
        spread = []
    try:
        total = db.fetch_all("SELECT * FROM residual_calibration_total ORDER BY bin_low")
    except Exception:
        total = []
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


# ======== Notification API ========

@app.get("/api/notifications")
async def get_notifications(limit: int = 30):
    try:
        from src.notifications.service import get_recent
        notifs = get_recent(limit=limit)
        return JSONResponse(notifs)
    except Exception:
        return JSONResponse([])


@app.get("/api/notifications/unread")
async def get_unread_count():
    try:
        from src.notifications.service import get_unread_count
        count = get_unread_count()
        return JSONResponse({"count": count})
    except Exception:
        return JSONResponse({"count": 0})


@app.post("/api/notifications/read/{notification_id}")
async def mark_notification_read(notification_id: int):
    from src.notifications.service import mark_read
    mark_read(notification_id)
    return JSONResponse({"status": "ok"})


@app.post("/api/notifications/read-all")
async def mark_all_notifications_read():
    from src.notifications.service import mark_all_read
    mark_all_read()
    return JSONResponse({"status": "ok"})


# ======== All-Star View ========

@app.get("/allstar", response_class=HTMLResponse)
async def allstar_page(request: Request):
    """All-Star Weekend analysis with 4 tabs matching desktop view."""
    mvp_candidates = [
        {"player": "LeBron James", "team": "LAL", "score": 42.1, "odds": "+800", "edge": 3.2},
        {"player": "Giannis Antetokounmpo", "team": "MIL", "score": 48.7, "odds": "+500", "edge": 5.1},
        {"player": "Luka Doncic", "team": "DAL", "score": 52.3, "odds": "+600", "edge": 4.8},
        {"player": "Jayson Tatum", "team": "BOS", "score": 38.9, "odds": "+1000", "edge": 2.1},
        {"player": "Nikola Jokic", "team": "DEN", "score": 50.2, "odds": "+700", "edge": 4.5},
        {"player": "Shai Gilgeous-Alexander", "team": "OKC", "score": 41.0, "odds": "+900", "edge": 3.0},
    ]
    three_pt_contestants = [
        {"player": "Stephen Curry", "team": "GSW", "three_pct": "42.8%", "three_pm": 5.1, "score": 88.5, "odds": "+350", "edge": 8.2},
        {"player": "Klay Thompson", "team": "DAL", "three_pct": "38.5%", "three_pm": 3.2, "score": 72.1, "odds": "+800", "edge": 3.1},
        {"player": "Buddy Hield", "team": "GSW", "three_pct": "40.1%", "three_pm": 3.8, "score": 76.4, "odds": "+600", "edge": 5.0},
        {"player": "Desmond Bane", "team": "MEM", "three_pct": "39.7%", "three_pm": 3.1, "score": 71.2, "odds": "+900", "edge": 2.8},
    ]
    rising_stars = [
        {"player": "Victor Wembanyama", "team": "SAS", "year": "2nd", "ppg": 24.2, "score": 85.0, "odds": "+300", "edge": 7.5},
        {"player": "Chet Holmgren", "team": "OKC", "year": "2nd", "ppg": 16.5, "score": 68.2, "odds": "+800", "edge": 3.0},
        {"player": "Brandon Miller", "team": "CHO", "year": "2nd", "ppg": 17.3, "score": 65.1, "odds": "+1000", "edge": 2.5},
    ]
    game_teams = [
        {"team": "Team East", "proj_score": 178.5, "win_prob": "52.3%", "spread": "-1.5", "odds": "-115", "edge": 2.1},
        {"team": "Team West", "proj_score": 175.0, "win_prob": "47.7%", "spread": "+1.5", "odds": "-105", "edge": 1.8},
    ]
    return templates.TemplateResponse("allstar.html", {
        "request": request,
        "active": "allstar",
        "mvp_candidates": mvp_candidates,
        "three_pt_contestants": three_pt_contestants,
        "rising_stars": rising_stars,
        "game_teams": game_teams,
    })


# ======== Team Colors API ========

@app.get("/api/regression/save")
async def sse_regression_save(name: str = Query("baseline")):
    """Save a regression test baseline."""
    def _run(callback):
        from src.analytics.regression_test import save_baseline
        return save_baseline(name, callback=callback)
    return StreamingResponse(
        _sse_generator(_run),
        media_type="text/event-stream",
    )


@app.get("/api/regression/compare")
async def sse_regression_compare(name: str = Query("baseline")):
    """Compare current predictions against a baseline."""
    def _run(callback):
        from src.analytics.regression_test import compare_to_baseline
        return compare_to_baseline(name, callback=callback)
    return StreamingResponse(
        _sse_generator(_run),
        media_type="text/event-stream",
    )


@app.get("/api/regression/list")
async def regression_list():
    from src.analytics.regression_test import list_baselines
    return JSONResponse(list_baselines())


@app.get("/api/regression/test-features")
async def regression_test_features():
    from src.analytics.regression_test import test_feature_extraction
    return JSONResponse(test_feature_extraction())


@app.get("/api/team-colors")
async def team_colors_api():
    try:
        from src.ui.widgets.nba_colors import TEAM_COLORS
        return JSONResponse({str(k): v for k, v in TEAM_COLORS.items()})
    except Exception:
        return JSONResponse({})


# ======== Startup ========

@app.on_event("startup")
async def startup():
    # init_db() and InjuryMonitor are handled by src.bootstrap.bootstrap(),
    # called from main.py before Uvicorn starts.  The calls below are
    # kept as idempotent fallbacks for direct `uvicorn src.web.app:app` usage.
    init_db()
    try:
        from src.notifications.injury_monitor import get_injury_monitor
        get_injury_monitor().start()   # no-op if already running
    except Exception as e:
        logger.warning(f"Injury monitor failed to start: {e}")
    logger.info("NBA Prediction System started")
