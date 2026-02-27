@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 launch_app.py
) else (
  python launch_app.py
)

if errorlevel 1 (
  echo.
  echo Buff-mini launcher failed. Review the messages above.
  pause
)

endlocal
