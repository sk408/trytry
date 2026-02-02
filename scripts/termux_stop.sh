#!/data/data/com.termux/files/usr/bin/bash
# Stop the NBA Betting Analytics server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.server.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping server (PID: $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "Server stopped."
    else
        echo "Server not running (stale PID file)."
        rm -f "$PID_FILE"
    fi
else
    echo "No PID file found. Server may not be running."
    echo "To find running server: pgrep -f 'python main.py'"
fi
