@echo off
setlocal
cd /d "%~dp0"
title MLB Market Intelligence v2.9 Setup

set "PYTHON_CMD="
where py >nul 2>nul && set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
  where python >nul 2>nul && set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
  echo Python 3 was not found.
  echo Install Python from https://www.python.org/downloads/windows/
  echo During installation, select "Add python.exe to PATH".
  goto :error
)

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating the app environment...
  %PYTHON_CMD% -m venv .venv || goto :error
  ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

echo Starting the local prediction page at http://127.0.0.1:8765
".venv\Scripts\python.exe" prediction_app.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Setup could not finish. The message above explains what is missing.
pause
exit /b 1
