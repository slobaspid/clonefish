@echo off
REM Double-click to play a 3+0 game against a clone in your browser (http://localhost:5001).
REM Close this window to stop the server.
cd /d "%~dp0"
set PYTHONPATH=%~dp0
start "" "http://localhost:5001"
"C:\Users\sloba\AppData\Local\Programs\Python\Python312\python.exe" scripts\clonefish_play.py --port 5001
