#!/data/data/com.termux/files/usr/bin/bash
# Termux setup script for NBA Betting Analytics
# Run once after cloning the repo to install dependencies and create venv

set -e

echo "=== NBA Betting Analytics - Termux Setup ==="

# Update package list
echo "[1/5] Updating package list..."
pkg update -y

# Install Python and required system packages
echo "[2/5] Installing Python and build tools..."
pkg install -y python git libxml2 libxslt

# Navigate to project directory (script location)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
echo "[3/5] Project directory: $PROJECT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "[4/5] Creating virtual environment..."
    python -m venv .venv
else
    echo "[4/5] Virtual environment already exists"
fi

# Activate venv and install dependencies
echo "[5/5] Installing Python dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the app:"
echo "  cd $PROJECT_DIR"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo ""
echo "Then open http://127.0.0.1:8000 in your browser."
echo ""
echo "Or use: ./scripts/termux_run.sh"
