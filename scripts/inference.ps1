# Quick Inference Script
# Attiva ambiente inference ed esegue test

Write-Host "🎯 INFERENCE MODE" -ForegroundColor Yellow

if (-not (Test-Path ".venv_inference")) {
    Write-Host "❌ Ambiente inference non trovato!" -ForegroundColor Red
    Write-Host "   Esegui: .\scripts\setup_dual_environments.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Attivazione ambiente inference..." -ForegroundColor Cyan
& .\.venv_inference\Scripts\Activate.ps1

Write-Host "Esecuzione test...`n" -ForegroundColor Green
python finetuning_workflow.py test $args

deactivate
Write-Host "`n✓ Test completato - ambiente disattivato" -ForegroundColor Green
