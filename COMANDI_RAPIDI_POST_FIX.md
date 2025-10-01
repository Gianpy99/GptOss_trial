# 🚀 Quick Commands - Riferimento Rapido Post-Fix

**Data**: 1 ottobre 2025  
**Status**: Tutto risolto e pronto all'uso ✅

---

## 🎯 Training Rapido (RACCOMANDATO per GTX 1660 SUPER)

### Lightweight Test (8-10 minuti)
```powershell
.\.venv_training\Scripts\Activate.ps1
python test_training_lightweight.py
```
**Output atteso**: Training completo in ~8-10 minuti con 20 esempi

---

## 🏗️ Training Development (30-40 minuti)

Modifica prima `test_training_fixed.py`:
```python
LIMIT = 50
EPOCHS = 2
MAX_LENGTH = 384
```

Poi esegui:
```powershell
.\.venv_training\Scripts\Activate.ps1
python test_training_fixed.py
```

---

## 🎯 Training Production (2-4 ore)

Modifica prima `test_training_fixed.py`:
```python
LIMIT = 200
EPOCHS = 2
BATCH_SIZE = 1
MAX_LENGTH = 512
```

Poi esegui:
```powershell
.\.venv_training\Scripts\Activate.ps1
python test_training_fixed.py
```

---

## 📤 Deploy & Test

```powershell
# Disattiva training env
deactivate

# Attiva inference env
.\.venv_inference\Scripts\Activate.ps1

# Deploy adapter su Ollama
python finetuning_workflow.py deploy --project f1_lightweight

# Test inference
python finetuning_workflow.py test --project f1_lightweight
```

---

## 🔍 Verifica Ambiente

```powershell
# Verifica GPU
.\.venv_training\Scripts\Activate.ps1
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Output atteso**:
```
CUDA: True
GPU: NVIDIA GeForce GTX 1660 SUPER
```

---

## 🐛 Debug Veloce

```powershell
# Verifica Python version
python --version

# Verifica path
where python

# Verifica import PyTorch
python -c "import torch; print(torch.__version__)"

# Verifica import Transformers
python -c "from transformers import AutoTokenizer; print('OK')"
```

---

## 📊 Confronto Configurazioni

| Comando | Tempo | Esempi | Uso |
|---------|-------|--------|-----|
| `python test_training_lightweight.py` | 8-10 min | 20 | Quick test, debug |
| `python test_training_fixed.py` (50 ex, 2 ep) | 30-40 min | 50 | Development |
| `python test_training_fixed.py` (50 ex, 3 ep) | 60-85 min | 50 | Standard |
| `python test_training_fixed.py` (200 ex, 2 ep) | 2-4 ore | 200 | Production |

---

## 🔒 Sicurezza Token (URGENTE)

1. Revoca il token HuggingFace:
   - Vai a: https://huggingface.co/settings/tokens
   - Trova il token compromesso e cliccca "Revoke"

2. Crea un nuovo token:
   - Clicca "New token"
   - Scope: Read (o Write se serve fine-tuning dei modelli su HF Hub)
   - Copia il token

3. Aggiorna il file `.env`:
   ```bash
   HF_TOKEN=hf_nuovo_token_qui
   ```

4. **MAI** committare il file `.env` o token in file tracciati!

---

## 📚 Documentazione Completa

- **`RIEPILOGO_COMPLETO.md`** - Sommario di tutto
- **`TRAINING_FIX_AND_OPTIMIZATION.md`** - Guida dettagliata fix & ottimizzazioni
- **`DUAL_ENVIRONMENT_SETUP.md`** - Setup ambienti Python
- **`QUICK_REFERENCE.md`** - Comandi rapidi giornalieri

---

## ✅ Checklist Prima del Training

- [ ] Ambiente `.venv_training` attivato
- [ ] CUDA disponibile (`torch.cuda.is_available() == True`)
- [ ] HF Token configurato (file `.env` presente con token valido)
- [ ] Parametri ottimizzati scelti (vedi tabella sopra)
- [ ] Spazio disco sufficiente (~2-5 GB)
- [ ] Nessun altro processo GPU in esecuzione

---

## 💡 Tips

### Se il training è troppo lento:
- Usa `test_training_lightweight.py` (8-10 min)
- Riduci `LIMIT`, `EPOCHS`, `MAX_LENGTH`
- Usa LoRA `r=8` invece di `r=16`

### Se hai crash OOM:
- Riduci `BATCH_SIZE` a 1
- Riduci `MAX_LENGTH` a 256
- Usa LoRA `r=4` o `r=8`

### Per migliorare la qualità:
- Aumenta `LIMIT` (più esempi)
- Aumenta `EPOCHS` (più iterazioni)
- Aumenta `MAX_LENGTH` (più contesto)

---

**Ultimo aggiornamento**: 1 ottobre 2025  
**Status**: ✅ Pronto per l'uso  
**Prossimo passo raccomandato**: Esegui `python test_training_lightweight.py` per validare che tutto funzioni!
