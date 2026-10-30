@echo off
setlocal
for %%I in ("%~dp0..\..\..\..\..") do set "PACKAGE_ROOT=%%~fI"
set "PYTHON=%PACKAGE_ROOT%\02_CONFIGURATION\portable_python_3_12\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" "%~dp0process_planetscope_standard_hedley.py" --input-zip "%PACKAGE_ROOT%\04_DATA\01_RAW\02_PLANET_DOVE\01_INDONESIA\01_TIDUNG\20251117_8420286\Planet_Dove_Indonesia_Tidung_20251117_SR_UDM2.zip" --output-dir "%PACKAGE_ROOT%\05_RESULTS\01_INTERMEDIATE\02_PLANET_DOVE_3M\01_INDONESIA\01_TIDUNG\20251117_8420286\01_GLINT_CORRECTION" --distance-m 300
pause
