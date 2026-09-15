@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Local environment missing. See README.md.
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -m sprayframegen %*
exit /b %errorlevel%
