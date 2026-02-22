#!/data/data/com.termux/files/usr/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Print an intro message
echo "=========================================="
echo "Starting NBA FastAPI Server..."
echo "Access it on your phone browser at:"
echo "http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server"
echo "=========================================="

# Start the application
python main.py
