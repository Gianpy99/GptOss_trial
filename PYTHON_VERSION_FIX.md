# ⚠️ PROBLEMA: Python 3.13 + PyTorch CUDA

## Situazione Attuale
- Python: **3.13.3** ❌
- PyTorch: Solo versione CPU (2.8.0+cpu)
- GPU: NVIDIA GTX 1660 (6GB) - **DISPONIBILE ma NON USATA**

## Il Problema
**PyTorch non supporta ancora Python 3.13!**  
Versioni supportate: Python 3.8, 3.9, 3.10, 3.11, 3.12

---

## ✅ SOLUZIONE 1: Crea Virtual Environment con Python 3.11 (CONSIGLIATO)

### Step 1: Installa Python 3.11
Scarica da: https://www.python.org/downloads/release/python-3110/

Oppure usa `pyenv` o `conda`

### Step 2: Crea Virtual Environment
```powershell
# Con Python 3.11 installato
py -3.11 -m venv .venv311

# Attiva
.\.venv311\Scripts\Activate.ps1

# Verifica
python --version  # Dovrebbe mostrare 3.11.x
```

### Step 3: Installa Dipendenze
```powershell
# PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Fine-tuning libs
pip install -r requirements-finetuning.txt
```

### Step 4: Verifica GPU
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Dovrebbe stampare:
```
CUDA: True
GPU: NVIDIA GeForce GTX 1660 Ti
```

### Step 5: Esegui Training
```powershell
python finetuning_workflow.py train `
  --dataset "Vadera007/Formula_1_Dataset" `
  --project "f1_expert" `
  --model "gemma3:4b" `
  --type "f1" `
  --epochs 1 `
  --batch-size 2 `
  --limit 50
```

**Tempo atteso con GPU**: 3-5 minuti! 🚀

---

## ✅ SOLUZIONE 2: Usa Conda (Python 3.11 + CUDA gestito automaticamente)

### Step 1: Installa Miniconda
Scarica: https://docs.conda.io/en/latest/miniconda.html

### Step 2: Crea Ambiente
```powershell
conda create -n finetuning python=3.11 -y
conda activate finetuning
```

### Step 3: Installa PyTorch con CUDA
```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Step 4: Installa Altre Dipendenze
```powershell
pip install transformers peft datasets accelerate bitsandbytes
pip install -r requirements.txt
```

### Step 5: Training
```powershell
python finetuning_workflow.py train --dataset "Vadera007/Formula_1_Dataset" --project "f1_expert" --model "gemma3:4b" --type "f1"
```

---

## ⚡ SOLUZIONE 3: Training su Computer con GPU Potente (Il Tuo Scenario)

Dato che:
- Computer A (training): Deve avere Python 3.11/3.12 + GPU
- Computer B (inference): Può usare Python 3.13 + solo Ollama

### Setup Computer Training
```powershell
# Su computer con GPU potente
py -3.11 -m venv .venv_training
.\.venv_training\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-finetuning.txt

# Training (5-10 minuti con GPU)
python finetuning_workflow.py train ...

# Output: finetuning_projects/f1_expert/ (~50MB)
```

### Transfer su Computer Inference
```powershell
# Comprimi
Compress-Archive -Path "finetuning_projects\f1_expert" -DestinationPath "f1_expert.zip"

# Trasferisci via USB/Network

# Deploy (su computer inference - Python 3.13 OK!)
python finetuning_workflow.py deploy --project f1_expert
```

---

## 🆘 SOLUZIONE TEMPORANEA: Continua con CPU (LENTO)

Se non puoi installare Python 3.11 adesso:

```powershell
# Training con CPU (30-60 minuti per epoch)
python finetuning_workflow.py train `
  --dataset "Vadera007/Formula_1_Dataset" `
  --project "f1_expert" `
  --model "gemma3:4b" `
  --type "f1" `
  --epochs 1 `
  --batch-size 1 `
  --limit 20  # Riduci esempi

# Lascialo girare in background
```

**Prestazioni**:
- Con CPU: 30-60 minuti
- Con GPU (dopo fix): 3-5 minuti ⚡

---

## 📊 Comparazione Soluzioni

| Soluzione | Tempo Setup | Velocità Training | Difficoltà |
|-----------|-------------|-------------------|------------|
| VirtualEnv 3.11 | 10 min | 10x più veloce | Facile ⭐⭐ |
| Conda | 15 min | 10x più veloce | Media ⭐⭐⭐ |
| Computer Training | 0 min | 10x+ più veloce | Facile ⭐ |
| CPU (temporaneo) | 0 min | Lento | Immediato ⭐ |

---

## 🎯 Raccomandazione

**Per te**: Usa **Soluzione 3** (Training su altro computer)

Perché:
- Computer inference: Python 3.13 OK (solo Ollama)
- Computer training: Python 3.11 + GPU potente
- Workflow già progettato per questo
- Adapter piccoli (~50MB) facili da trasferire

**Su computer training**:
```powershell
# Installa Python 3.11
py -3.11 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
py -3.11 -m pip install -r requirements-finetuning.txt

# Training veloce con GPU
py -3.11 finetuning_workflow.py train ...
```

**Su questo computer (inference)**:
- Rimani con Python 3.13
- Usa solo OllamaWrapper (nessun fine-tuning)
- Deploy adapter creati sull'altro computer

---

**Vuoi procedere con quale soluzione?** 🤔
