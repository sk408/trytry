@echo off
title NBA Betting Analytics - Desktop
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python desktop.py
pause
