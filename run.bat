@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Локальное окружение не найдено. Выполните подготовку по README.md.
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -m sprayframegen %*
exit /b %errorlevel%
