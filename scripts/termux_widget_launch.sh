#!/data/data/com.termux/files/usr/bin/bash
# Termux:Widget launcher script
# One-tap to start server and open browser
#
# Setup (run once in Termux):
#   mkdir -p ~/.shortcuts
#   chmod +x ~/trytry/scripts/termux_widget_launch.sh
#   ln -sf ~/trytry/scripts/termux_widget_launch.sh ~/.shortcuts/NBA-Analytics
#
# Then add the Termux:Widget to your home screen and select "NBA-Analytics"

# Project configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.server.pid"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/server.log"
APP_URL="http://127.0.0.1:8000"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

# Function to check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0
        else
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

# Function to start server
start_server() {
    if [ ! -d ".venv" ]; then
        echo "Error: Virtual environment not found."
        echo "Run termux_setup.sh first."
        termux-toast "Setup required! Run termux_setup.sh"
        exit 1
    fi
    
    source .venv/bin/activate
    
    echo "$(date): Starting server..." >> "$LOG_FILE"
    nohup python main.py >> "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"
    
    # Wait briefly for server to start
    sleep 2
    
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server started (PID: $SERVER_PID)"
        return 0
    else
        echo "Failed to start server"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Main logic
if is_running; then
    echo "Server already running"
    termux-toast "Server running - Opening browser"
else
    echo "Starting server..."
    termux-toast "Starting server..."
    if start_server; then
        termux-toast "Server started - Opening browser"
    else
        termux-toast "Failed to start server!"
        exit 1
    fi
fi

# Open browser
echo "Opening browser: $APP_URL"
termux-open-url "$APP_URL"
