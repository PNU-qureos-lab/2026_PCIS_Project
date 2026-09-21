@echo off
setlocal
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer first.
  pause
  exit /b 1
)
python -c "import numpy, rasterio, PIL" >nul 2>&1
if errorlevel 1 (
  echo Installing the portable workflow dependencies...
  python -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)
python run_complete_example.py --input-dir ..\01_INPUT --output-dir ..\04_GENERATED_OUTPUT --scene 20260808_S2A
if errorlevel 1 (
  echo Workflow failed. Review the error shown above.
  pause
  exit /b 1
)
echo.
echo Complete. Results are in 04_GENERATED_OUTPUT.
pause
