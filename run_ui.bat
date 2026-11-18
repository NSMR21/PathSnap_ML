@echo off
REM Laptop 2: serve the evaluator UI on http://localhost:8501
REM Then open it and type laptop 1's address in the API box, e.g. http://192.168.1.23:8000
cd /d "%~dp0"
start "" http://localhost:8501
python -m http.server 8501 --directory ui
