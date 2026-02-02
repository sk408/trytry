# NBA Betting Analytics (Web)

This project now runs as a lightweight FastAPI web app so it can be used on Android via Termux. It reuses the existing analytics, data sync, and SQLite database layers; only the UI and entrypoint changed.

## Android/Termux Installation

### Prerequisites

1. Install [Termux](https://f-droid.org/packages/com.termux/) from F-Droid (Google Play version is outdated)
2. Optionally install these Termux add-ons for shortcuts and auto-start:
   - [Termux:Widget](https://f-droid.org/packages/com.termux.widget/) - home screen shortcuts
   - [Termux:Boot](https://f-droid.org/packages/com.termux.boot/) - auto-start on device boot

### Quick Setup (Automated)

```bash
# Clone the repo
git clone https://github.com/sk408/trytry.git
cd trytry

# Run the setup script (installs Python, creates venv, installs deps)
bash scripts/termux_setup.sh
```

The setup script uses Termux's pre-built packages for numpy, pandas, and matplotlib (via `pkg install`), which is **much faster** than compiling from source with pip. Setup typically takes 2-5 minutes instead of 30-45 minutes.

### Quick Setup (Manual)

1) Install Python and dev tools:
```bash
pkg update
pkg install python git libxml2 libxslt
```

2) Create a virtualenv and install deps:
```bash
cd /path/to/trytry
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the App

### Option 1: Using the run script
```bash
./scripts/termux_run.sh
```

With options:
```bash
# Run in background
./scripts/termux_run.sh --background

# Run with logging
./scripts/termux_run.sh --log server.log

# Both
./scripts/termux_run.sh -b --log server.log
```

### Option 2: Manual start
```bash
source .venv/bin/activate
python main.py
```

Open the UI in your mobile browser at `http://127.0.0.1:8000`.

### Stopping the Server

If running in foreground: Press `Ctrl+C`

If running in background:
```bash
./scripts/termux_stop.sh
```

## Home Screen Shortcut (Termux:Widget)

Create a home screen shortcut to start/stop the app with one tap:

```bash
# Create shortcuts directory
mkdir -p ~/.shortcuts

# Link the service script
ln -s "$(pwd)/scripts/termux_service.sh" ~/.shortcuts/NBA-Analytics

# Optionally add stop shortcut
ln -s "$(pwd)/scripts/termux_stop.sh" ~/.shortcuts/NBA-Stop
```

Then add the Termux:Widget to your home screen and tap "NBA-Analytics" to start.

## Auto-Start on Boot (Termux:Boot)

Start the server automatically when your device boots:

```bash
# Create boot directory
mkdir -p ~/.termux/boot

# Link the service script
ln -s "$(pwd)/scripts/termux_service.sh" ~/.termux/boot/nba-analytics.sh
```

The server will start in the background on boot. Logs are written to `logs/server.log`.

## Features

- Dashboard: trigger data sync, injury sync, and build injury history.
- Live: fetch live scores and show betting recommendations.
- Players: roster list plus injured-player impact table.
- Matchups: pick teams to generate projected spread/total; view upcoming games.
- Schedule: next 14 days of games (run sync first).
- Accuracy: run backtests with optional injury adjustments.
- Admin: reset the SQLite database and re-initialize schema.

## Notes

- Database location is managed by `src/database/db.py` (defaults to `data/nba_analytics.db`).
- Long-running sync/backtest operations run inside the API process; on Termux, keep the session open while they complete.
- No front-end build step is required (plain HTML/Jinja + CSS).
