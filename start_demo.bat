@echo off
rem One-click launcher: starts the model server, then opens http://127.0.0.1:8000/
cd /d "%~dp0"

set "PY=..\minimind-master\.venv-demo\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] venv python not found:
    echo   %PY%
    echo Check that minimind-master sits next to this folder.
    pause
    exit /b 1
)

echo Python: %PY%
echo Starting MiniMind demo, model loads in about 10 seconds...
echo.

start "" cmd /c "timeout /t 12 >nul & start http://127.0.0.1:8000/"
"%PY%" server.py

echo.
echo Server stopped. Press any key to close.
pause >nul
