@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=python"
where py >nul 2>&1
if %errorlevel%==0 (
    py -3.10 -c "import sys" >nul 2>&1
    if %errorlevel%==0 set "PYTHON=py -3.10"
)

if not exist ".venv\Scripts\python.exe" (
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :fail
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :fail
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

if not exist "data\UTM 96-98.xlsx" (
    echo ERROR: data\UTM 96-98.xlsx was not found.
    goto :fail
)

if not exist outputs mkdir outputs

python src\motorcycle_crash_pipeline.py --data "data\UTM 96-98.xlsx" --output "outputs\main_analysis" --mode full
if errorlevel 1 goto :fail

python src\supplementary_analyses.py --pipeline "src\motorcycle_crash_pipeline.py" --primary-output "outputs\main_analysis" --output "outputs\supplementary_analyses"
if errorlevel 1 goto :fail

python src\rebuild_shap_interaction_figures.py --pipeline "src\motorcycle_crash_pipeline.py" --primary-output "outputs\main_analysis" --figure-output "outputs\interaction_figures"
if errorlevel 1 goto :fail

echo COMPLETE
exit /b 0

:fail
echo Run failed.
exit /b 1
