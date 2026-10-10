@echo off
setlocal
cd /d "%~dp0"

rem Usage: run.bat [train|predict] [main.py options]
rem   run.bat                                  -> train AAPL (both targets)
rem   run.bat predict --ticker AAPL
rem   run.bat train --ticker MSFT --target return --epochs 50

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.9+ from python.org first.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || exit /b 1
)

rem Reinstall only when requirements.txt changed since the last successful install
set "STAMP=.venv\.requirements.stamp"
set "NEEDS_INSTALL=1"
if exist "%STAMP%" (
    fc /b requirements.txt "%STAMP%" >nul 2>nul && set "NEEDS_INSTALL="
)
if defined NEEDS_INSTALL (
    echo Installing dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
    copy /y requirements.txt "%STAMP%" >nul
)

".venv\Scripts\python.exe" main.py %*
exit /b %errorlevel%
