@echo off
setlocal
cd /d "%~dp0"

echo Opening the MLB prediction app...
call START_APP.bat
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo The app did not start. Read the message above.
pause
exit /b 1
