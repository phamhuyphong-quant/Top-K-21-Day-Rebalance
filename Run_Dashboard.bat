@echo off
title VN100 Dashboard Runner
echo ========================================
echo Starting VN100 Trading Dashboard...
echo ========================================

cd /d "%~dp0"

:: 1. Check if the virtual environment folder (.venv) exists
if not exist .venv (
    echo [1/3] Creating a new virtual environment...
    python -m venv .venv
)

:: 2. Activate and show library status
echo [2/3] Syncing Python libraries with requirements.txt...
call .venv\Scripts\activate

:: Removing the -q flag shows the download/install progress
pip install -r requirements.txt

echo.
echo Current Environment Libraries:
echo ----------------------------------------
:: This shows the user exactly what is installed in the .venv
pip list
echo ----------------------------------------
echo.

:: 3. Launch the app
echo [3/3] Launching Streamlit...
python -m streamlit run src/app.py

echo.
echo Dashboard session ended.
pause