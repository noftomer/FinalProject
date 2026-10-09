@echo off
cd /d "%~dp0"
call run.bat train --ticker AAPL --epochs 50 --seq-len 60
pause
