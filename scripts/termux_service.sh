#!/data/data/com.termux/files/usr/bin/bash
# Termux:Boot / Termux:Widget service script for NBA Betting Analytics
#
# This script can be used with:
#   1. Termux:Boot - auto-start on device boot
#   2. Termux:Widget - home screen shortcut
#   3. Termux:Services - background service management
#
# Installation options:
#
# Option A - Termux:Boot (auto-start on boot):
#   mkdir -p ~/.termux/boot
#   ln -s /path/to/trytry/scripts/termux_service.sh ~/.termux/boot/nba-analytics.sh
#
# Option B - Termux:Widget (home screen shortcut):
#   mkdir -p ~/.shortcuts
#   ln -s /path/to/trytry/scripts/termux_service.sh ~/.shortcuts/NBA-Analytics
#
# Option C - Termux:Services (sv-style service):
#   mkdir -p ~/.termux/sv/nba-analytics
#   ln -s /path/to/trytry/scripts/termux_service.sh ~/.termux/sv/nba-analytics/run

set -e

# Project configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$PROJECT_DIR/.server.pid"
LOG_FILE="$LOG_DIR/server.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Server already running (PID: $OLD_PID)"
        echo "To stop: kill $OLD_PID"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found."
    echo "Run termux_setup.sh first."
    exit 1
fi

source .venv/bin/activate

# Start server
echo "$(date): Starting NBA Betting Analytics server..." >> "$LOG_FILE"
echo "Server starting..."
echo "URL: http://127.0.0.1:8000"
echo "Log: $LOG_FILE"

# Run in foreground if called by sv (Termux:Services)
# Otherwise run in background
if [ -n "$SVDIR" ]; then
    # Running under Termux:Services (sv/runit)
    exec python main.py 2>&1
else
    # Running standalone (boot/widget) - background it
    nohup python main.py >> "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"
    echo "Server started in background (PID: $SERVER_PID)"
    echo ""
    echo "To stop the server:"
    echo "  kill $SERVER_PID"
    echo "  # or"
    echo "  kill \$(cat $PID_FILE)"
fi
