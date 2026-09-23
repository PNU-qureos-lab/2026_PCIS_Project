@echo off
setlocal
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python run_one_scene.py
if errorlevel 1 exit /b 1
python render_step_results.py
if errorlevel 1 exit /b 1
echo Complete. Results are in ..\results_step_by_step\20260808_S2A
pause
