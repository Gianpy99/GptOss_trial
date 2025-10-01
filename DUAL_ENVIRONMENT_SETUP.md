# 🚀 Setup Dual Environment - Training & Inference

## Strategia: Due Ambienti Virtuali Separati

### 📦 Ambiente 1: Training (Python 3.11 + GPU)
- Fine-tuning con PyTorch + CUDA
- Creazione adapter LoRA
- Usa GPU per velocità

### 📦 Ambiente 2: Inference (Python 3.13 + CPU)
- OllamaWrapper per inference
- Deploy adapter
- Test modelli

---

## ⚡ Setup Automatico (15 minuti)

### STEP 1: Installa Python 3.11

Se non hai Python 3.11, scarica da:
🔗 https://www.python.org/downloads/release/python-31110/

**Windows Installer**: `python-3.11.10-amd64.exe`

Durante installazione:
- ✅ Add Python 3.11 to PATH
- ✅ Install pip

Verifica:
```powershell
py -3.11 --version  # Python 3.11.10
py -3.13 --version  # Python 3.13.3
```

### STEP 2: Crea Ambienti Virtuali

Esegui questo script PowerShell:

```powershell
# Salva come: setup_environments.ps1

Write-Host "🚀 Creazione Dual Environment Setup..." -ForegroundColor Cyan

# Ambiente Training (Python 3.11)
Write-Host "`n📦 Ambiente 1: TRAINING (Python 3.11 + GPU)" -ForegroundColor Yellow
if (Test-Path ".venv_training") {
    Write-Host "⚠️  .venv_training già esistente, eliminazione..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv_training
}

py -3.11 -m venv .venv_training
Write-Host "✓ Virtual environment creato: .venv_training" -ForegroundColor Green

# Attiva ambiente training
.\.venv_training\Scripts\Activate.ps1

Write-Host "`n📥 Installazione PyTorch con CUDA..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "`n📥 Installazione dipendenze fine-tuning..." -ForegroundColor Cyan
pip install transformers peft datasets accelerate bitsandbytes scikit-learn scipy
pip install -r requirements.txt

Write-Host "`n✅ Test GPU..." -ForegroundColor Cyan
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

deactivate

# Ambiente Inference (Python 3.13)
Write-Host "`n`n📦 Ambiente 2: INFERENCE (Python 3.13)" -ForegroundColor Yellow
if (Test-Path ".venv_inference") {
    Write-Host "⚠️  .venv_inference già esistente, eliminazione..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv_inference
}

py -3.13 -m venv .venv_inference
Write-Host "✓ Virtual environment creato: .venv_inference" -ForegroundColor Green

# Attiva ambiente inference
.\.venv_inference\Scripts\Activate.ps1

Write-Host "`n📥 Installazione dipendenze inference..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`n✅ Test OllamaWrapper..." -ForegroundColor Cyan
python -c "from src.ollama_wrapper import OllamaWrapper; print('✓ OllamaWrapper importato correttamente')"

deactivate

Write-Host "`n`n🎉 SETUP COMPLETATO!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "Ambienti creati:" -ForegroundColor Cyan
Write-Host "  1. .venv_training  (Python 3.11 + GPU)" -ForegroundColor White
Write-Host "  2. .venv_inference (Python 3.13)" -ForegroundColor White
Write-Host "`nComandi rapidi:" -ForegroundColor Cyan
Write-Host "  Training:  .\.venv_training\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  Inference: .\.venv_inference\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "================================" -ForegroundColor Green
```

Esegui:
```powershell
# Crea file
notepad setup_environments.ps1
# Copia il contenuto sopra e salva

# Esegui setup
powershell -ExecutionPolicy Bypass -File setup_environments.ps1
```

---

## 🎯 Uso Quotidiano

### Per TRAINING (Fine-Tuning)

```powershell
# Attiva ambiente training
.\.venv_training\Scripts\Activate.ps1

# Verifica GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Training con GPU
python finetuning_workflow.py train `
  --dataset "Vadera007/Formula_1_Dataset" `
  --project "f1_expert" `
  --model "gemma3:4b" `
  --type "f1" `
  --epochs 3 `
  --batch-size 4 `
  --limit 100

# Output: finetuning_projects/f1_expert/

# Disattiva
deactivate
```

### Per INFERENCE (Test Modelli)

```powershell
# Attiva ambiente inference
.\.venv_inference\Scripts\Activate.ps1

# Test modello base
python -c "from src.ollama_wrapper import OllamaWrapper; w = OllamaWrapper('gemma3:4b'); print(w.chat('Test'))"

# Deploy adapter
python finetuning_workflow.py deploy --project f1_expert

# Test confronto
python finetuning_workflow.py test --project f1_expert

# Disattiva
deactivate
```

---

## 📂 Struttura Progetto

```
Ollama_wrapper/
├── .venv_training/          # Python 3.11 + PyTorch CUDA
├── .venv_inference/         # Python 3.13 + OllamaWrapper
├── .env                     # HF_TOKEN (non committato)
├── .gitignore               # Ignora .venv_*
├── finetuning_workflow.py   # Script principale
├── requirements.txt         # Dipendenze base
├── requirements-finetuning.txt
├── finetuning_projects/
│   └── f1_expert/           # Adapter creato (~50MB)
│       ├── training_data.json
│       ├── metadata.json
│       └── adapter/
└── src/
    └── ollama_wrapper/
```

---

## 🖥️ Computer A (Sviluppo) - Uso Entrambi

### Workflow Tipico

```powershell
# Mattina: Training
.\.venv_training\Scripts\Activate.ps1
python finetuning_workflow.py train --dataset "my_data.json" --project "my_model"
deactivate

# Pomeriggio: Test Inference
.\.venv_inference\Scripts\Activate.ps1
python finetuning_workflow.py test --project "my_model"
deactivate
```

---

## 🖥️ Computer B (Inference) - Solo Ambiente Inference

Sul computer di inference:

```powershell
# Setup una volta sola
py -3.13 -m venv .venv_inference
.\.venv_inference\Scripts\Activate.ps1
pip install -r requirements.txt

# Copia adapter da Computer A
# (via USB/Network: finetuning_projects/my_model/)

# Deploy e usa
python finetuning_workflow.py deploy --project "my_model"
python finetuning_workflow.py test --project "my_model"
```

---

## 📋 Script Helper (Opzionale)

Crea `train.ps1`:
```powershell
# Quick training script
.\.venv_training\Scripts\Activate.ps1
python finetuning_workflow.py train $args
deactivate
```

Crea `inference.ps1`:
```powershell
# Quick inference script
.\.venv_inference\Scripts\Activate.ps1
python finetuning_workflow.py test $args
deactivate
```

Uso:
```powershell
# Training
.\train.ps1 --dataset "data.json" --project "model1"

# Inference
.\inference.ps1 --project "model1"
```

---

## ✅ Vantaggi Approccio Dual-Env

1. **Isolamento Perfetto**
   - Training non interferisce con inference
   - Versioni Python diverse nello stesso progetto

2. **Flessibilità**
   - Switch rapido tra ambienti
   - Testa subito dopo training

3. **Portabilità**
   - Computer A: entrambi ambienti
   - Computer B: solo inference
   - Adapter piccoli (~50MB) trasferibili

4. **Sicurezza**
   - `.venv_*` in `.gitignore`
   - Nessuna dipendenza globale
   - Clean uninstall: `rm -r .venv_*`

---

## 🔧 Aggiorna .gitignore

Aggiungi a `.gitignore`:
```gitignore
# Virtual environments
.venv_training/
.venv_inference/
.venv/
venv/
env/
```

---

## 🚀 Prossimi Step

1. **Esegui setup script** (15 min)
2. **Test training** con GPU
3. **Test inference** con Ollama
4. **Trasferisci adapter** su Computer B

**Tutto pronto per workflow professionale training/inference!** 🎉
