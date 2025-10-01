# 🚀 GEMMA 3 4B - ISTRUZIONI RAPIDE

## ⚡ 3 Step per iniziare

### STEP 1: Accetta Licenza
🔗 https://huggingface.co/google/gemma-3-4b-it

Clicca **"Agree and access repository"**

### STEP 2: Verifica Token
Token già in `.env`: ✅
```
HF_TOKEN=#################
```

### STEP 3: Esegui
```powershell
python finetuning_workflow.py train --dataset "Vadera007/Formula_1_Dataset" --project "f1_expert" --model "gemma3:4b" --type "f1" --epochs 1 --batch-size 2 --limit 50
```

---

## ✅ Se Funziona Vedi:
```
✓ Loaded HF token from .env file
✓ Downloaded 50 rows
✓ Created 60 training examples
🔥 STARTING FINE-TUNING
📦 Model: gemma3:4b → google/gemma-3-4b-it
⏳ Loading model...
```

## ❌ Se NON Funziona:

**403 Forbidden** → Non hai accettato licenza su Step 1  
**401 Unauthorized** → Token errato  
**Out of memory** → Usa `--batch-size 1`

---

**Durata**: 5-15 minuti  
**Output**: `finetuning_projects/f1_expert/` (~50MB adapter)
