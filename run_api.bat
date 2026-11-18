@echo off
REM Laptop 1: start the PathSnap API on port 8000 (reachable from the network).
REM Optional: set PATHSNAP_TRIPS_DIR=C:\path\to\evaluator_test   before running.
cd /d "%~dp0"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
