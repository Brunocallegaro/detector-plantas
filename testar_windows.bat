@echo off
REM Executa todos os testes automatizados usando a venv do projeto.
chcp 65001 >nul
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Rode primeiro iniciar_windows.bat para criar a venv.
    pause
    exit /b 1
)
venv\Scripts\python.exe -m pip install -r requirements-dev.txt
venv\Scripts\python.exe -m pytest -v
pause
