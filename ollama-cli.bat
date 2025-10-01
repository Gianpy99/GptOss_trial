@echo off
REM Ollama Wrapper CLI - Windows Launcher
REM Usage: ollama-cli.bat [command] [args...]

SET VENV_PATH=%~dp0.venv_training
SET PYTHON_SCRIPT=%~dp0ollama_cli.py

REM Attiva virtual environment
call "%VENV_PATH%\Scripts\activate.bat"

REM Esegui CLI
python "%PYTHON_SCRIPT%" %*

REM Deattiva venv
call deactivate
