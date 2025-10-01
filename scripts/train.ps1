# Quick Training Script
# Attiva ambiente training ed esegue fine-tuning

Write-Host "🔥 TRAINING MODE" -ForegroundColor Yellow

if (-not (Test-Path ".venv_training")) {
    Write-Host "❌ Ambiente training non trovato!" -ForegroundColor Red
    Write-Host "   Esegui: .\scripts\setup_dual_environments.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Attivazione ambiente training..." -ForegroundColor Cyan
& .\.venv_training\Scripts\Activate.ps1

Write-Host "Esecuzione training...`n" -ForegroundColor Green
python finetuning_workflow.py train $args

deactivate
Write-Host "`n✓ Training completato - ambiente disattivato" -ForegroundColor Green
