@echo off
cd /d "%~dp0"
call run.bat train --ticker AAPL --refresh
pause
