# Script PowerShell per avviare UI GPU
# Salva come: start_ui_gpu.ps1

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  🏎️ F1 Expert UI - GPU Accelerated" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Attiva ambiente
Write-Host "📦 Attivazione ambiente training..." -ForegroundColor Yellow
& .\.venv_training\Scripts\Activate.ps1

# Verifica CUDA
Write-Host "🔍 Verifica CUDA..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
Write-Host ""

# Avvia UI
Write-Host "🚀 Avvio UI (caricamento modello richiede 30-60 sec)..." -ForegroundColor Green
Write-Host "⏳ Attendi che si apra il browser..." -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 Per fermare: Ctrl+C" -ForegroundColor Magenta
Write-Host ""

python ui_gpu.py
