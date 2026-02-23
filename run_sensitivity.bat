@echo off
echo Running Sensitivity Analysis...
echo Usage: run_sensitivity.bat --param [name] --min [val] --max [val] --steps [num]
echo    Or: run_sensitivity.bat --all --steps [num]
python -m src.analytics.sensitivity %*