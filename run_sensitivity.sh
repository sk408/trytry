#!/bin/bash
echo "Running Sensitivity Analysis..."
echo "Usage: ./run_sensitivity.sh --param [name] --min [val] --max [val] --steps [num]"
echo "   Or: ./run_sensitivity.sh --all --steps [num]"
python -m src.analytics.sensitivity "$@"