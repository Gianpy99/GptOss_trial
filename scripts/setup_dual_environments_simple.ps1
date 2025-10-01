# Setup Dual Environment - Training & Inference
# Versione semplificata senza caratteri Unicode problematici

Write-Host ""
Write-Host "========================================"
Write-Host "  DUAL ENVIRONMENT SETUP"
Write-Host "========================================"
Write-Host ""

# Verifica Python 3.11
Write-Host "Verifica versioni Python..."
$py311_check = py -3.11 --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRORE: Python 3.11 non trovato!"
    Write-Host "Scarica da: https://www.python.org/downloads/"
    exit 1
}
Write-Host "OK Python 3.11: $py311_check"

$py313_check = py -3.13 --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK Python 3.13: $py313_check"
    $hasPy313 = $true
} else {
    Write-Host "Python 3.13 non trovato (opzionale)"
    $hasPy313 = $false
}

Write-Host ""

# ============================================
# AMBIENTE 1: TRAINING (Python 3.11 + GPU)
# ============================================

Write-Host "========================================"
Write-Host "  AMBIENTE 1: TRAINING (Python 3.11)"
Write-Host "========================================"
Write-Host ""

if (Test-Path ".venv_training") {
    Write-Host "Rimozione .venv_training esistente..."
    Remove-Item -Recurse -Force .venv_training
}

Write-Host "Creazione virtual environment..."
py -3.11 -m venv .venv_training
Write-Host "OK Virtual environment creato: .venv_training"
Write-Host ""

Write-Host "Attivazione ambiente..."
& .\.venv_training\Scripts\Activate.ps1

Write-Host ""
Write-Host "Installazione PyTorch con CUDA 12.1..."
Write-Host "(Questo richiedera circa 2-3GB download, 5-10 minuti)"
Write-Host ""

pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRORE installazione PyTorch"
    deactivate
    exit 1
}
Write-Host "OK PyTorch CUDA installato"
Write-Host ""

Write-Host "Installazione dipendenze fine-tuning..."
pip install --quiet transformers peft datasets accelerate bitsandbytes scikit-learn scipy

if (Test-Path "requirements.txt") {
    Write-Host "Installazione requirements.txt..."
    pip install --quiet -r requirements.txt
}

Write-Host ""
Write-Host "Test configurazione GPU..."
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"

Write-Host ""
Write-Host "Ambiente TRAINING configurato!"
Write-Host ""

deactivate

# ============================================
# AMBIENTE 2: INFERENCE (Python 3.13)
# ============================================

if ($hasPy313) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  AMBIENTE 2: INFERENCE (Python 3.13)"
    Write-Host "========================================"
    Write-Host ""

    if (Test-Path ".venv_inference") {
        Write-Host "Rimozione .venv_inference esistente..."
        Remove-Item -Recurse -Force .venv_inference
    }

    Write-Host "Creazione virtual environment..."
    py -3.13 -m venv .venv_inference
    Write-Host "OK Virtual environment creato: .venv_inference"
    Write-Host ""

    Write-Host "Attivazione ambiente..."
    & .\.venv_inference\Scripts\Activate.ps1

    Write-Host ""
    Write-Host "Installazione dipendenze base..."
    
    if (Test-Path "requirements.txt") {
        pip install --quiet -r requirements.txt
    } else {
        pip install --quiet requests
    }

    Write-Host ""
    Write-Host "Test OllamaWrapper..."
    python -c "from src.ollama_wrapper import OllamaWrapper; print('OK OllamaWrapper importato')"

    Write-Host ""
    Write-Host "Ambiente INFERENCE configurato!"
    Write-Host ""

    deactivate
}

# ============================================
# RIEPILOGO FINALE
# ============================================

Write-Host ""
Write-Host "========================================"
Write-Host "  SETUP COMPLETATO!"
Write-Host "========================================"
Write-Host ""

Write-Host "Ambienti creati:"
Write-Host "  .venv_training  (Python 3.11 + PyTorch CUDA)"
if ($hasPy313) {
    Write-Host "  .venv_inference (Python 3.13 + OllamaWrapper)"
}

Write-Host ""
Write-Host "Comandi rapidi:"
Write-Host ""
Write-Host "  Per TRAINING:"
Write-Host "     .\.venv_training\Scripts\Activate.ps1"
Write-Host "     python finetuning_workflow.py train --dataset '...' --project '...'"
Write-Host ""

if ($hasPy313) {
    Write-Host "  Per INFERENCE:"
    Write-Host "     .\.venv_inference\Scripts\Activate.ps1"
    Write-Host "     python finetuning_workflow.py test --project '...'"
    Write-Host ""
}

Write-Host "Esempio completo F1:"
Write-Host '  .\.venv_training\Scripts\Activate.ps1'
Write-Host '  python finetuning_workflow.py train \'
Write-Host '    --dataset "Vadera007/Formula_1_Dataset" \'
Write-Host '    --project "f1_expert" \'
Write-Host '    --model "gemma3:4b" \'
Write-Host '    --type "f1" \'
Write-Host '    --epochs 3 \'
Write-Host '    --batch-size 4 \'
Write-Host '    --limit 100'
Write-Host '  deactivate'
Write-Host ""

Write-Host "Documentazione completa: DUAL_ENVIRONMENT_SETUP.md"
Write-Host "========================================"
Write-Host ""
