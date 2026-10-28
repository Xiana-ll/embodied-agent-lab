@echo off
rem ============================================================
rem  One-click launcher: starts the model server and opens the chat page.
rem  If the venv path below does not exist, it falls back to "python".
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

set "PY=C:\Users\24437\Desktop\11111\实验室\minimind-master\.venv-demo\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Starting MiniMind demo (loading model, about 10 seconds)...
echo.

rem open the browser after a delay, so it does not hit a not-ready port
start "" cmd /c "timeout /t 10 >nul & start http://127.0.0.1:8000/"

"%PY%" server.py

echo.
echo Server stopped. Press any key to close this window.
pause >nul
