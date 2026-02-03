# College Basketball Analytics

A betting analytics platform for NCAA college basketball (men's and women's).

This project has two UIs:
- **Desktop (PySide6)** - Full-featured desktop app for Windows/Mac/Linux
- **Web (FastAPI)** - Lightweight mobile-friendly web app for Android via Termux

Both share the same analytics, data sync, and SQLite database.

## Key Differences from NBA Version

| Feature | NBA | College |
|---------|-----|---------|
| Game length | 48 min (4 quarters) | 40 min (2 halves) |
| Team count | 30 | ~1,100+ (all divisions) |
| Avg scoring | ~112 PPG | ~70-75 PPG |
| Data source | nba_api library | ESPN API + CBBpy |
| Home court adv | ~3.0 pts | ~2.5 pts |

## Data Loading Strategy

Unlike the NBA version which syncs all 30 teams upfront, college basketball uses **day-ahead loading**:

1. Each morning: Fetch today's scheduled games from ESPN
2. For each game: Load team rosters and recent player stats
3. Cache data for 24 hours

This approach handles the large number of college teams efficiently.

## Desktop Version (Windows/Mac/Linux)

```bash
# Install desktop dependencies
pip install -r requirements-desktop.txt

# Run desktop app
python desktop.py
```

## Web/Mobile Version (Android via Termux)

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

# Switch to college basketball branch
git checkout college-basketball

# Run the setup script (installs Python, creates venv, installs deps)
bash scripts/termux_setup.sh
```

The setup script adds the Termux User Repository (tur-repo) and installs pre-built numpy, pandas, and matplotlib packages. No compilation needed - setup takes just 2-5 minutes.

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
ln -s "$(pwd)/scripts/termux_service.sh" ~/.shortcuts/CBB-Analytics

# Optionally add stop shortcut
ln -s "$(pwd)/scripts/termux_stop.sh" ~/.shortcuts/CBB-Stop
```

Then add the Termux:Widget to your home screen and tap "CBB-Analytics" to start.

## Auto-Start on Boot (Termux:Boot)

Start the server automatically when your device boots:

```bash
# Create boot directory
mkdir -p ~/.termux/boot

# Link the service script
ln -s "$(pwd)/scripts/termux_service.sh" ~/.termux/boot/cbb-analytics.sh
```

The server will start in the background on boot. Logs are written to `logs/server.log`.

## Features

- **Dashboard**: Trigger data sync (day-ahead loading), injury sync, and build injury history
- **Live**: Fetch live scores with halftime tracking and betting recommendations
- **Players**: Roster list plus injured-player impact table
- **Matchups**: Pick teams to generate projected spread/total; supports neutral site games
- **Schedule**: Today's games and tomorrow's schedule
- **Accuracy**: Run backtests with optional injury adjustments
- **Admin**: Reset the SQLite database and re-initialize schema

## Data Sources

- **Primary**: ESPN API (free, no key required)
  - Scoreboard, team rosters, box scores, odds
  - Supports both men's and women's college basketball
- **Supplementary**: CBBpy library
  - Detailed play-by-play data
  - Additional box score stats

## Supported Leagues

- Men's College Basketball (NCAA D1, D2, D3)
- Women's College Basketball (NCAA D1, D2, D3)

Toggle between leagues in the app settings.

## Notes

- Database location: `data/college_analytics.db`
- Long-running sync/backtest operations run inside the API process; on Termux, keep the session open while they complete
- No front-end build step is required (plain HTML/Jinja + CSS)
- College basketball uses 20-minute halves (not 12-minute quarters)

## Future: March Madness

March Madness support is planned as a separate feature branch with:
- Bracket structure (68 teams, regions, seeds)
- Single-elimination prediction adjustments
- Neutral site handling for tournament games
- Upset probability metrics
