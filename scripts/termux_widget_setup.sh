#!/data/data/com.termux/files/usr/bin/bash
# Setup Termux:Widget shortcut for NBA Betting Analytics
#
# Prerequisites:
#   1. Install Termux:Widget from F-Droid
#   2. Run termux_setup.sh first to set up the project
#
# After running this script:
#   1. Long-press on your home screen
#   2. Add Widget -> Termux:Widget
#   3. Select "NBA-Analytics" from the list

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SHORTCUT_DIR="$HOME/.shortcuts"
SHORTCUT_NAME="NBA-Analytics"
LAUNCH_SCRIPT="$SCRIPT_DIR/termux_widget_launch.sh"

echo "=== Termux Widget Setup ==="
echo ""

# Check if Termux:API is available (for termux-toast, termux-open-url)
if ! command -v termux-toast &> /dev/null; then
    echo "[!] Termux:API not installed. Installing..."
    pkg install -y termux-api
fi

# Create shortcuts directory
if [ ! -d "$SHORTCUT_DIR" ]; then
    echo "[1/3] Creating ~/.shortcuts directory..."
    mkdir -p "$SHORTCUT_DIR"
else
    echo "[1/3] ~/.shortcuts directory exists"
fi

# Make launch script executable
echo "[2/3] Making launch script executable..."
chmod +x "$LAUNCH_SCRIPT"

# Create symlink
echo "[3/3] Creating widget shortcut..."
ln -sf "$LAUNCH_SCRIPT" "$SHORTCUT_DIR/$SHORTCUT_NAME"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To add the widget to your home screen:"
echo "  1. Long-press on your home screen"
echo "  2. Select 'Widgets'"
echo "  3. Find 'Termux:Widget' and drag it to your home screen"
echo "  4. Select '$SHORTCUT_NAME' from the list"
echo ""
echo "Tapping the widget will:"
echo "  - Start the server (if not running)"
echo "  - Open the app in your browser"
echo ""
