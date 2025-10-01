# Setup Dual Environment - Training & Inference
# Crea due ambienti virtuali separati per workflow completo

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  DUAL ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Verifica Python 3.11 e 3.13
Write-Host "Verifica versioni Python..." -ForegroundColor Yellow

try {
    $py311 = py -3.11 --version 2>&1
    Write-Host "✓ Python 3.11: $py311" -ForegroundColor Green
} catch {
    Write-Host "❌ Python 3.11 non trovato!" -ForegroundColor Red
    Write-Host "   Scarica da: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

try {
    $py313 = py -3.13 --version 2>&1
    Write-Host "✓ Python 3.13: $py313" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Python 3.13 non trovato (opzionale)" -ForegroundColor Yellow
}

Write-Host "`n" -NoNewline

# ============================================
# AMBIENTE 1: TRAINING (Python 3.11 + GPU)
# ============================================

Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  AMBIENTE 1: TRAINING (Python 3.11)" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

if (Test-Path ".venv_training") {
    Write-Host "⚠️  Rimozione .venv_training esistente..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv_training
}

Write-Host "Creazione virtual environment..." -ForegroundColor Cyan
py -3.11 -m venv .venv_training
Write-Host "✓ Virtual environment creato: .venv_training`n" -ForegroundColor Green

Write-Host "Attivazione ambiente..." -ForegroundColor Cyan
& .\.venv_training\Scripts\Activate.ps1

Write-Host "`n📥 Installazione PyTorch con CUDA 12.1..." -ForegroundColor Cyan
Write-Host "   (Questo richiederà ~2-3GB download, 5-10 minuti)`n" -ForegroundColor Gray

pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ PyTorch CUDA installato`n" -ForegroundColor Green
} else {
    Write-Host "❌ Errore installazione PyTorch`n" -ForegroundColor Red
    deactivate
    exit 1
}

Write-Host "📥 Installazione dipendenze fine-tuning..." -ForegroundColor Cyan
pip install --quiet transformers peft datasets accelerate bitsandbytes scikit-learn scipy

if (Test-Path "requirements.txt") {
    Write-Host "📥 Installazione requirements.txt..." -ForegroundColor Cyan
    pip install --quiet -r requirements.txt
}

Write-Host "`n✅ Test configurazione GPU..." -ForegroundColor Cyan
python -c "import torch; print(f'  CUDA Available: {torch.cuda.is_available()}'); print(f'  GPU Device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('  GPU non rilevata')"

Write-Host "`n✓ Ambiente TRAINING configurato!`n" -ForegroundColor Green

deactivate

# ============================================
# AMBIENTE 2: INFERENCE (Python 3.13)
# ============================================

if (Test-Path "py.exe") {
    $hasPy313 = $true
} else {
    Write-Host "`n⚠️  Python 3.13 non trovato - skip ambiente inference" -ForegroundColor Yellow
    Write-Host "   (Puoi crearlo manualmente dopo)" -ForegroundColor Gray
    $hasPy313 = $false
}

if ($hasPy313) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "  AMBIENTE 2: INFERENCE (Python 3.13)" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Yellow

    if (Test-Path ".venv_inference") {
        Write-Host "⚠️  Rimozione .venv_inference esistente..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force .venv_inference
    }

    Write-Host "Creazione virtual environment..." -ForegroundColor Cyan
    py -3.13 -m venv .venv_inference
    Write-Host "✓ Virtual environment creato: .venv_inference`n" -ForegroundColor Green

    Write-Host "Attivazione ambiente..." -ForegroundColor Cyan
    & .\.venv_inference\Scripts\Activate.ps1

    Write-Host "`n📥 Installazione dipendenze base..." -ForegroundColor Cyan
    
    if (Test-Path "requirements.txt") {
        pip install --quiet -r requirements.txt
    } else {
        pip install --quiet requests
    }

    Write-Host "`n✅ Test OllamaWrapper..." -ForegroundColor Cyan
    python -c "try:`n    from src.ollama_wrapper import OllamaWrapper`n    print('  OK OllamaWrapper importato')`nexcept Exception as e:`n    print(f'  Errore: {e}')"

    Write-Host "`n✓ Ambiente INFERENCE configurato!`n" -ForegroundColor Green

    deactivate
}

# ============================================
# RIEPILOGO FINALE
# ============================================

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  🎉 SETUP COMPLETATO!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "Ambienti creati:" -ForegroundColor Cyan
Write-Host "  📦 .venv_training  (Python 3.11 + PyTorch CUDA)" -ForegroundColor White
if ($hasPy313) {
    Write-Host "  📦 .venv_inference (Python 3.13 + OllamaWrapper)" -ForegroundColor White
}

Write-Host "`nComandi rapidi:" -ForegroundColor Cyan
Write-Host "`n  🔥 Per TRAINING:" -ForegroundColor Yellow
Write-Host "     .\.venv_training\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "     python finetuning_workflow.py train --dataset '...' --project '...'`n" -ForegroundColor Gray

if ($hasPy313) {
    Write-Host "  🎯 Per INFERENCE:" -ForegroundColor Yellow
    Write-Host "     .\.venv_inference\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "     python finetuning_workflow.py test --project '...'`n" -ForegroundColor Gray
}

Write-Host "Esempio completo:" -ForegroundColor Cyan
Write-Host @"

  # Training
  .\.venv_training\Scripts\Activate.ps1
  python finetuning_workflow.py train ``
    --dataset "Vadera007/Formula_1_Dataset" ``
    --project "f1_expert" ``
    --model "gemma3:4b" ``
    --type "f1" ``
    --epochs 3 ``
    --batch-size 4 ``
    --limit 100
  deactivate
"@ -ForegroundColor Gray

Write-Host "`n📚 Documentazione completa: DUAL_ENVIRONMENT_SETUP.md" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Green
