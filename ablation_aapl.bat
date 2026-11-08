@echo off
setlocal
cd /d "%~dp0"

rem Usage: ablation_aapl.bat [ticker] [seeds]
rem   ablation_aapl.bat              -> AAPL, 5 seeds
rem   ablation_aapl.bat MSFT 3

set "TICKER=%~1"
if not defined TICKER set "TICKER=AAPL"
set "SEEDS=%~2"
if not defined SEEDS set "SEEDS=5"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run run.bat once first to create it.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m src.ablation --ticker %TICKER% --seeds %SEEDS%
pause
