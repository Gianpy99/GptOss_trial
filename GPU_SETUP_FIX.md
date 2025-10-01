# 🔧 Fix GPU Support - Installa PyTorch con CUDA

## Problema Identificato
PyTorch installato è **CPU-only** (`2.8.0+cpu`)  
La tua GPU (6GB VRAM) non viene utilizzata ❌

## ✅ Soluzione: Reinstalla PyTorch con CUDA

### Step 1: Disinstalla PyTorch CPU
```powershell
pip uninstall torch torchvision torchaudio -y
```

### Step 2: Installa PyTorch con CUDA 11.8 o 12.1

**Opzione A: CUDA 11.8 (più compatibile)**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Opzione B: CUDA 12.1 (più recente)**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

⚠️ **Nota**: Scegli la versione CUDA che hai installato sul sistema.  
Verifica con: `nvidia-smi` (guarda in alto "CUDA Version: X.X")

### Step 3: Verifica GPU Funzionante
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Dovresti vedere:
```
CUDA: True
GPU: NVIDIA GeForce RTX ... (o la tua GPU)
```

### Step 4: Reinstalla bitsandbytes (per 4-bit quantization)
```powershell
pip uninstall bitsandbytes -y
pip install bitsandbytes
```

---

## 🚀 Dopo l'Installazione

### Test Completo
```powershell
python -c "import torch; import bitsandbytes; print('✓ PyTorch CUDA:', torch.cuda.is_available()); print('✓ GPU:', torch.cuda.get_device_name(0)); print('✓ bitsandbytes OK')"
```

### Esegui Fine-Tuning (Ora userà la GPU!)
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

---

## 📊 Prestazioni Attese

### Con CPU (attuale)
- Caricamento modello: ~2 minuti
- Training 1 epoch (50 esempi): **30-60 minuti** 🐌
- RAM richiesta: ~16GB

### Con GPU 6GB (dopo fix)
- Caricamento modello: ~20 secondi
- Training 1 epoch (50 esempi): **3-5 minuti** 🚀
- VRAM richiesta: ~5GB (con 4-bit quantization)

---

## 🔍 Troubleshooting

### "CUDA: False" dopo reinstallazione
1. Verifica driver NVIDIA aggiornati
2. Riavvia terminale/IDE
3. Verifica CUDA installato: `nvidia-smi`

### "Out of memory" durante training
GPU 6GB è sufficiente con 4-bit quantization.  
Se problemi:
```powershell
--batch-size 1  # Riduci batch size
--limit 30      # Riduci esempi
```

### "bitsandbytes not compiled with CUDA"
```powershell
pip uninstall bitsandbytes -y
pip install bitsandbytes --no-cache-dir
```

---

## 💡 Alternative (se CUDA non si installa)

### Opzione 1: Training su altro computer con GPU
Il workflow è progettato per questo:
1. Training su computer A (con GPU potente)
2. Adapter (~50MB) su USB
3. Deploy su computer B (inference)

### Opzione 2: Google Colab (GPU gratis)
Usa notebook Colab con GPU T4 gratis:
1. Upload script su Colab
2. Training con GPU cloud
3. Download adapter

### Opzione 3: CPU con modello più piccolo
Usa Phi-2 (2.7B) invece di Gemma (4B):
```powershell
--model "phi2"  # Più veloce su CPU
```

---

**Consiglio**: Installa CUDA PyTorch ora, sarai ~10x più veloce! 🚀
