@echo off
chcp 65001 >nul
cd /d "%~dp0"

:STYLEID_LOOP
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_styleid.ps1" -Interactive
set "STYLEID_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%STYLEID_EXIT_CODE%"=="0" (
    echo StyleID failed with exit code %STYLEID_EXIT_CODE%.
)

set "STYLEID_AGAIN="
set /p "STYLEID_AGAIN=Run another style transfer? [Y/n]: "
if /I "%STYLEID_AGAIN%"=="N" goto STYLEID_END
if /I "%STYLEID_AGAIN%"=="NO" goto STYLEID_END
if /I "%STYLEID_AGAIN%"=="Q" goto STYLEID_END

echo.
goto STYLEID_LOOP

:STYLEID_END
echo Finished.
pause
exit /b 0

