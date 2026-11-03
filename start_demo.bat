@echo off
rem One-click launcher. Starts the model server and opens the chat page.
rem The venv lives under a Chinese-named folder, so this script finds it by
rem structure instead of hardcoding the path (cmd mangles non-ASCII paths).
cd /d "%~dp0"

set "PY="
for /d %%D in ("..\*") do (
    if exist "%%D\minimind-master\.venv-demo\Scripts\python.exe" set "PY=%%D\minimind-master\.venv-demo\Scripts\python.exe"
)

if not defined PY (
    echo [ERROR] venv python not found.
    echo Expected: ..\*\minimind-master\.venv-demo\Scripts\python.exe
    pause
    exit /b 1
)

echo Found python: %PY%
echo Starting MiniMind demo, model loads in about 10 seconds...
echo.

start "" cmd /c "timeout /t 12 >nul & start http://127.0.0.1:8000/"
"%PY%" server.py

echo.
echo Server stopped. Press any key to close.
pause >nul
