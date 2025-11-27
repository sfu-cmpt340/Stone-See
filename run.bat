@echo off
REM KidneyNet - Windows Batch Runner
REM This is a simple wrapper for the Python runner

echo Starting KidneyNet...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.7+ from https://www.python.org/
    pause
    exit /b 1
)

REM Run the Python script
python run.py

pause

