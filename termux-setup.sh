#!/data/data/com.termux/files/usr/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Termux Setup for NBA App..."

# 1. Update Termux packages
echo "Updating Termux packages..."
pkg update -y
pkg upgrade -y

# 2. Install required system dependencies
echo "Installing Python and dependencies..."
pkg install -y python clang make libffi libzmq freetype libpng pkg-config

# Optional: Try to install math libraries via pkg (if available in Termux repo)
# If they fail, pip will pick them up later.
echo "Installing pre-built math libraries if available..."
pkg install -y python-numpy python-pandas matplotlib || echo "Could not install via pkg, will fall back to pip."

# 3. Create and activate a virtual environment
echo "Setting up Python virtual environment..."
python -m venv --system-site-packages venv
source venv/bin/activate

# 4. Upgrade pip and wheel
echo "Upgrading pip..."
pip install --upgrade pip wheel

# 5. Install Termux-compatible requirements
echo "Installing Python requirements..."
pip install -r requirements-termux.txt

echo "=========================================="
echo "Setup complete! ✅"
echo "You can now run the app using: ./termux-run.sh"
echo "=========================================="
