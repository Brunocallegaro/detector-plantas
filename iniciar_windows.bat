@echo off
REM ============================================================
REM  Detector de Doencas em Plantas - inicializacao no Windows
REM  1a vez: cria a venv e instala as bibliotecas (alguns minutos)
REM  Depois: abre direto a API + interface em http://localhost:8000
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 run_pipeline.py %*
) else (
    python run_pipeline.py %*
)
pause
