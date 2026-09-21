@echo off
setlocal enabledelayedexpansion
if exist ".venv\Scripts\python.exe" goto :run_app
echo.
echo ============================================================
echo  SprayFrameGen: lokalnoe okruzenie ne naydeno
echo ============================================================
echo.
echo Dlya podgotovki okruzeniya vipolnite odin raz:
echo.
echo   py -3.14 -m venv .venv
echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Dlya oflayn-podgotovki (bez interneta) zaranee skachayte
echo Windows x64 wheels dlya Python 3.14 i ustanovite:
echo.
echo   .venv\Scripts\python.exe -m pip install --no-index --find-links wheels -r requirements.txt
echo.
echo Podrobnosti sm. v README.md
echo ============================================================
exit /b 1
:run_app
".venv\Scripts\python.exe" -X utf8 -m sprayframegen %*
exit /b %errorlevel%