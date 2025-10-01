# Deploy Adapter Script
# Attiva ambiente inference e deploy adapter

Write-Host "🚀 DEPLOY MODE" -ForegroundColor Yellow

if (-not (Test-Path ".venv_inference")) {
    Write-Host "❌ Ambiente inference non trovato!" -ForegroundColor Red
    Write-Host "   Esegui: .\scripts\setup_dual_environments.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Attivazione ambiente inference..." -ForegroundColor Cyan
& .\.venv_inference\Scripts\Activate.ps1

Write-Host "Deploy adapter...`n" -ForegroundColor Green
python finetuning_workflow.py deploy $args

deactivate
Write-Host "`n✓ Deploy completato - ambiente disattivato" -ForegroundColor Green
