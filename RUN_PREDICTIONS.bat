@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating the Python environment...
  py -3 -m venv .venv || goto :error
  ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

".venv\Scripts\python.exe" run_predictions.py %*
if errorlevel 1 goto :error

echo.
echo Done. Open output\EDGE_PICKS.md to see the card.
exit /b 0

:error
echo.
echo Prediction run failed. Read the message above for the missing input or setup step.
exit /b 1

