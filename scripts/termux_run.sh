#!/data/data/com.termux/files/usr/bin/bash
# Termux run script for NBA Betting Analytics
# Starts the FastAPI server with optional logging

set -e

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse arguments
LOG_FILE=""
BACKGROUND=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        --background|-b)
            BACKGROUND=true
            shift
            ;;
        --help|-h)
            echo "Usage: termux_run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --log FILE       Write output to log file"
            echo "  --background, -b Run in background"
            echo "  --help, -h       Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Run termux_setup.sh first."
    exit 1
fi

echo "Starting NBA Betting Analytics..."
echo "Server: http://127.0.0.1:8000"
echo "Press Ctrl+C to stop"
echo ""

if [ "$BACKGROUND" = true ]; then
    if [ -n "$LOG_FILE" ]; then
        nohup python main.py > "$LOG_FILE" 2>&1 &
        echo "Running in background. PID: $!"
        echo "Log file: $LOG_FILE"
    else
        nohup python main.py > /dev/null 2>&1 &
        echo "Running in background. PID: $!"
    fi
    echo "To stop: kill $!"
elif [ -n "$LOG_FILE" ]; then
    python main.py 2>&1 | tee "$LOG_FILE"
else
    python main.py
fi
