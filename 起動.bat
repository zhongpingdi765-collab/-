@echo off
chcp 65001 > nul
cd /d "%~dp0"
where py >nul 2>&1 && set PY=py || set PY=python
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv
)
call .venv\Scripts\python -m pip install -r requirements.txt
call .venv\Scripts\python main.py serve
pause
