@echo off
setlocal
cd /d "%~dp0\.."
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -X utf8 codigo\compilar_pdf.py
) else (
    python -X utf8 codigo\compilar_pdf.py
)
if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo compilar el PDF. Revisa el mensaje anterior.
    pause
    exit /b 1
)
echo.
echo [OK] PDF generado en informe\reporte_tarea1.pdf
pause
