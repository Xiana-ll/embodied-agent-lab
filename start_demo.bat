@echo off
rem One-click launcher: starts the model server.
rem The server opens the browser itself once the port is listening,
rem so this window is the only console window you get.
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
echo Starting MiniMind demo. The browser opens by itself when ready.
echo Close this window or press Ctrl+C to stop the server.
echo.

"%PY%" server.py

echo.
echo Server stopped. Press any key to close.
pause >nul
