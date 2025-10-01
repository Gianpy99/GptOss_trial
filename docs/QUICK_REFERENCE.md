# 🚀 QUICK REFERENCE - Dual Environment

## Setup Iniziale (Una Volta Sola)

```powershell
# 1. Assicurati di avere Python 3.11 installato
py -3.11 --version

# 2. Esegui setup automatico
.\scripts\setup_dual_environments.ps1

# 3. Attendi ~10-15 minuti (download PyTorch CUDA)
```

---

## Comandi Rapidi

### Training (GPU)
```powershell
# Metodo 1: Script automatico
.\scripts\train.ps1 --dataset "Vadera007/Formula_1_Dataset" --project "f1_expert" --model "gemma3:4b" --type "f1" --epochs 3

# Metodo 2: Manuale
.\.venv_training\Scripts\Activate.ps1
python finetuning_workflow.py train --dataset "..." --project "..."
deactivate
```

### Inference (Test)
```powershell
# Metodo 1: Script automatico
.\scripts\inference.ps1 --project "f1_expert"

# Metodo 2: Manuale
.\.venv_inference\Scripts\Activate.ps1
python finetuning_workflow.py test --project "f1_expert"
deactivate
```

### Deploy Adapter
```powershell
# Metodo 1: Script automatico
.\scripts\deploy.ps1 --project "f1_expert"

# Metodo 2: Manuale
.\.venv_inference\Scripts\Activate.ps1
python finetuning_workflow.py deploy --project "f1_expert"
deactivate
```

---

## Workflow Tipico

### 1️⃣ Training con Dataset Custom
```powershell
# Crea file: my_data.json
# [{"instruction": "Q1", "output": "A1"}, ...]

.\scripts\train.ps1 `
  --dataset "my_data.json" `
  --project "my_model" `
  --model "gemma3:4b" `
  --type "qa" `
  --epochs 3 `
  --batch-size 4
```

**Output**: `finetuning_projects/my_model/` (~50MB)

### 2️⃣ Test Immediato
```powershell
.\scripts\inference.ps1 --project "my_model"
```

### 3️⃣ Trasferimento su Altro Computer
```powershell
# Comprimi adapter
Compress-Archive -Path "finetuning_projects\my_model" -DestinationPath "my_model.zip"

# Trasferisci via USB/Network/Cloud
```

---

## Su Computer di Inference (Solo Python 3.13)

```powershell
# Setup una volta
py -3.13 -m venv .venv_inference
.\.venv_inference\Scripts\Activate.ps1
pip install -r requirements.txt

# Decomprimi adapter ricevuto
Expand-Archive "my_model.zip" -Destination "finetuning_projects\"

# Deploy
python finetuning_workflow.py deploy --project "my_model"

# Test con Ollama
python finetuning_workflow.py test --project "my_model"
```

---

## Esempi Pratici

### F1 Dataset (Demo)
```powershell
.\scripts\train.ps1 `
  --dataset "Vadera007/Formula_1_Dataset" `
  --project "f1_expert" `
  --model "gemma3:4b" `
  --type "f1" `
  --epochs 3 `
  --limit 100
```

### Custom QA Dataset
```powershell
.\scripts\train.ps1 `
  --dataset "my_questions.json" `
  --project "qa_expert" `
  --model "gemma3:4b" `
  --type "qa" `
  --epochs 5
```

### Hugging Face Dataset
```powershell
.\scripts\train.ps1 `
  --dataset "username/dataset-name" `
  --project "hf_expert" `
  --model "gemma3:4b" `
  --type "generic"
```

---

## Verifica Ambienti

### Check Training Environment
```powershell
.\.venv_training\Scripts\Activate.ps1
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
deactivate
```

### Check Inference Environment
```powershell
.\.venv_inference\Scripts\Activate.ps1
python -c "from src.ollama_wrapper import OllamaWrapper; print('OK')"
deactivate
```

---

## Parametri Training

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--dataset` | (required) | Path o nome HF dataset |
| `--project` | (required) | Nome progetto |
| `--model` | `gemma3:4b` | Modello Ollama |
| `--type` | `generic` | Tipo: `f1`, `qa`, `generic` |
| `--epochs` | `3` | Numero epoch |
| `--batch-size` | `4` | Batch size (GPU 6GB: 2-4) |
| `--limit` | `100` | Max righe dataset |

---

## Troubleshooting

### "Ambiente non trovato"
```powershell
.\scripts\setup_dual_environments.ps1
```

### "Out of memory"
```powershell
.\scripts\train.ps1 ... --batch-size 1 --limit 50
```

### "CUDA not available"
Ricontrolla setup:
```powershell
.\.venv_training\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__)"  # Deve mostrare +cu121
```

---

## File Structure

```
Ollama_wrapper/
├── .venv_training/       # Python 3.11 + GPU
├── .venv_inference/      # Python 3.13
├── scripts/
│   ├── setup_dual_environments.ps1  # Setup iniziale
│   ├── train.ps1                    # Quick training
│   ├── inference.ps1                # Quick test
│   └── deploy.ps1                   # Quick deploy
├── finetuning_projects/
│   └── [project_name]/              # Adapter (~50MB)
└── finetuning_workflow.py           # Main script
```

---

**Pronto per iniziare!** 🚀

```powershell
# Step 1: Setup
.\scripts\setup_dual_environments.ps1

# Step 2: Training
.\scripts\train.ps1 --dataset "Vadera007/Formula_1_Dataset" --project "f1_test" --type "f1"

# Step 3: Test
.\scripts\inference.ps1 --project "f1_test"
```
