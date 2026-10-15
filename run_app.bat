@echo off
title VoxShield - Synthetic Voice Detection
echo ============================================================
echo   VoxShield - Synthetic Voice Fraud Detection
echo   Information Security Project - SLIIT
echo ============================================================
echo.
echo [1/2] Installing dependencies...
python -m pip install -r requirements.txt --quiet
echo.
echo [2/2] Starting the API and dashboard on http://127.0.0.1:8000
start "" http://127.0.0.1:8000
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
pause
