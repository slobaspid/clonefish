@echo off
REM clonefish UCI engine: VEGETAL's clone. Add this file as a UCI engine in any chess GUI or lichess-bot.
set ROOT=%~dp0..
set PYTHONPATH=%ROOT%
"C:\Users\sloba\AppData\Local\Programs\Python\Python312\python.exe" "%ROOT%\scripts\clonefish_uci.py" --player VEGETAL --pgn "%ROOT%\data\lichess_5k\VEGETAL.pgn.zst" --clone "%ROOT%\clones\ftval_VEGETAL_bucket.pt"
