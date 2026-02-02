#!/data/data/com.termux/files/usr/bin/bash
# Termux setup script for NBA Betting Analytics
# Run once after cloning the repo to install dependencies and create venv

set -e

echo "=== NBA Betting Analytics - Termux Setup ==="

# Update package list
echo "[1/7] Updating package list..."
pkg update -y

# Install Python and required system packages
echo "[2/7] Installing Python and build tools..."
pkg install -y python git libxml2 libxslt

# Add Termux User Repository for pre-built pandas
echo "[3/7] Adding Termux User Repository (tur-repo)..."
pkg install -y tur-repo

# Install pre-built scientific packages (avoids compilation)
echo "[4/7] Installing pre-built numpy, pandas, matplotlib..."
pkg install -y python-numpy python-pandas matplotlib

# Navigate to project directory (script location)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
echo "[5/7] Project directory: $PROJECT_DIR"

# Create virtual environment with system site-packages access
if [ ! -d ".venv" ]; then
    echo "[6/7] Creating virtual environment (with system packages)..."
    python -m venv --system-site-packages .venv
else
    echo "[6/7] Virtual environment already exists"
fi

# Activate venv and install remaining dependencies (pure Python, fast)
echo "[7/7] Installing remaining Python dependencies..."
source .venv/bin/activate
pip install --upgrade pip

# Install lightweight packages (no compilation needed)
pip install fastapi uvicorn jinja2 python-multipart nba_api requests beautifulsoup4

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
