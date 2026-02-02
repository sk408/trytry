param(
    [string]$Python = ".\\.venv\\Scripts\\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    Write-Error "Python executable not found at $Python. Ensure the virtual env is created."
}

$specArgs = @(
    "-m",
    "PyInstaller",
    "--noconfirm",
    "--clean",
    "--name", "nba-betting-analytics",
    "--windowed",
    "main.py",
    "--add-data", "data;data"
)

& $Python $specArgs
