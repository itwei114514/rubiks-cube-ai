@echo off
rem ===== start the Rubik's Cube web UI =====
cd /d "%~dp0"
set "PY=F:\VScode_program\Agent\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo Starting web UI at http://127.0.0.1:8000/ ...
start "" http://127.0.0.1:8000/
"%PY%" server.py
pause