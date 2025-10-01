# Ollama Wrapper CLI - PowerShell Launcher
# Usage: .\ollama-cli.ps1 [command] [args...]

$VenvPath = Join-Path $PSScriptRoot ".venv_training"
$PythonScript = Join-Path $PSScriptRoot "ollama_cli.py"

# Attiva virtual environment
& "$VenvPath\Scripts\Activate.ps1"

# Esegui CLI con tutti gli argomenti
python $PythonScript $args

# Exit code
$ExitCode = $LASTEXITCODE
deactivate
exit $ExitCode
