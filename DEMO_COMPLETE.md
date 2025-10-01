# 🎯 Demo: Esperienza CLI Completa

## ✅ Hai ora un sistema completo!

### 📦 Cosa è stato creato:

1. **`ollama_cli.py`** - CLI Python principale
2. **`ollama-cli.bat`** - Launcher Windows (batch)
3. **`ollama-cli.ps1`** - Launcher PowerShell
4. **`ui_multi_model.py`** - Web UI con selector modelli
5. **`quick_train.py`** - Training script ottimizzato
6. **`combine_datasets.py`** - Utility per unire dataset
7. **`CLI_GUIDE.md`** - Guida completa

---

## 🚀 Quick Start

### 1️⃣ Lista tutto disponibile

```powershell
.\ollama-cli.ps1 list
```

**Output atteso:**
```
Ollama Models:
  • f1-expert:latest
  • gemma3:4b

Datasets:
  • combined_f1_tolkien_data.json
  • f1_training_data.json
  • tolkien_training_data.json

Training Projects:
  • f1_expert_fixed [adapter✓]
  • hybrid_expert
```

---

### 2️⃣ Avvia UI Multi-Modello

```powershell
.\ollama-cli.ps1 ui
```

**Features UI:**
- ✅ Selector modelli dinamico (dropdown)
- ✅ Switch modello senza riavviare
- ✅ Controllo temperature e tokens
- ✅ Example buttons per test rapidi
- ✅ Model info display
- ✅ Clear chat
- ✅ Copy responses

**Accesso:** http://localhost:7860

---

### 3️⃣ Training Nuovo Modello (esempio completo)

```powershell
# Step 1: Crea/combina dataset
python combine_datasets.py

# Step 2: Training rapido (50 esempi)
.\ollama-cli.ps1 train --dataset combined_f1_tolkien_data.json --project my_hybrid --limit 50 --epochs 2

# Step 3: Merge adapter (automatico se non usi --no-merge)
# .\ollama-cli.ps1 merge --project my_hybrid

# Step 4: Deploy su Ollama (automatico se non usi --no-deploy)
# .\ollama-cli.ps1 deploy --project my_hybrid --name my-hybrid --test

# Step 5: Test in UI
.\ollama-cli.ps1 ui --model my-hybrid
```

---

### 4️⃣ Pipeline Completa (UN COMANDO)

```powershell
# Fa tutto automaticamente: train + merge + deploy + UI
.\ollama-cli.ps1 pipeline --dataset combined_f1_tolkien_data.json --project awesome_model

# Workflow automatico:
# 1. ✅ Training (50 esempi, 2 epochs, ~10-15 min)
# 2. ✅ Merge adapter con base model (~3-5 min)
# 3. ✅ Deploy su Ollama (~2 min)
# 4. ✅ Launch Web UI
#
# Premi ENTER quando chiesto per avviare la UI
```

---

## 🎨 UI Multi-Modello - Features

### Screenshot Mentale della UI:

```
╔════════════════════════════════════════════════════════════╗
║  🏎️ Ollama Multi-Model Chat                               ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  💬 Conversation                    ⚙️ Model Settings     ║
║  ┌────────────────────────┐         ┌─────────────────┐   ║
║  │ User: Who is Gandalf? │         │ Select Model:   │   ║
║  │                        │         │ [f1-expert  ▼]  │   ║
║  │ Assistant: Gandalf is  │         │                 │   ║
║  │ a wizard sent to...    │         │ 🔄 Refresh      │   ║
║  └────────────────────────┘         │                 │   ║
║                                      │ Model: f1-expert│   ║
║  ┌────────────────────────┐         │ Size: 8.6 GB    │   ║
║  │ Your message here...   │         │                 │   ║
║  └────────────────────────┘         │ Temperature: 0.7│   ║
║                                      │ [────●────────] │   ║
║  [🗑️ Clear] [💡 F1] [💡 Tolkien]    │                 │   ║
║                                      │ Max Tokens: 250 │   ║
║                                      └─────────────────┘   ║
╚════════════════════════════════════════════════════════════╝
```

### Come usare:

1. **Seleziona modello** dal dropdown
2. **Regola temperature** (0.0 = preciso, 2.0 = creativo)
3. **Scrivi messaggio** e clicca Send
4. **Cambia modello** on-the-fly senza reload
5. **Usa example buttons** per test rapidi

---

## 📊 Workflow Tipici

### Scenario A: Primo Training

```powershell
# 1. Prepara dataset
python combine_datasets.py

# 2. Usa pipeline completa
.\ollama-cli.ps1 pipeline --dataset combined_f1_tolkien_data.json

# 3. (Aspetta 15-20 min)

# 4. Testa nella UI che si apre automaticamente
```

### Scenario B: Re-deploy Esistente

```powershell
# 1. Lista progetti
.\ollama-cli.ps1 list --type projects

# 2. Deploy
.\ollama-cli.ps1 deploy --project f1_expert_fixed --name f1-v2 --test

# 3. UI
.\ollama-cli.ps1 ui --model f1-v2
```

### Scenario C: Sviluppo Iterativo

```powershell
# Ciclo di sviluppo rapido

# Train veloce
.\ollama-cli.ps1 train --dataset my_data.json --limit 20 --epochs 1 --no-deploy

# Test locale (senza Ollama)
python quick_train.py my_data.json test_model 20

# Se OK, merge e deploy
.\ollama-cli.ps1 merge --project test_model
.\ollama-cli.ps1 deploy --project test_model --test

# UI per valutazione
.\ollama-cli.ps1 ui
```

---

## 🎯 Comandi Essenziali

```powershell
# Lista tutto
.\ollama-cli.ps1 list

# Train
.\ollama-cli.ps1 train --dataset my_data.json

# UI
.\ollama-cli.ps1 ui

# Pipeline completa
.\ollama-cli.ps1 pipeline --dataset my_data.json

# Help
.\ollama-cli.ps1 --help
.\ollama-cli.ps1 train --help
```

---

## 💡 Tips

### Alias PowerShell (Optional)

Aggiungi a `$PROFILE`:

```powershell
function ollama-cli {
    & "C:\Development\Ollama_wrapper\ollama-cli.ps1" $args
}

# Poi usa:
ollama-cli list
ollama-cli ui
```

### Quick Test

```powershell
# Test veloce (5 min)
.\ollama-cli.ps1 train --dataset combined_f1_tolkien_data.json --limit 10 --epochs 1 --project quicktest
```

### Backup

```powershell
# Esporta modello
ollama show my-model --modelfile > backup.Modelfile

# Lista con status
.\ollama-cli.ps1 list --type projects
```

---

## ✅ Checklist Verifica

- [x] CLI funziona: `.\ollama-cli.ps1 list`
- [x] UI si apre: `.\ollama-cli.ps1 ui`
- [x] Modelli visibili nella UI dropdown
- [x] Training completa senza errori
- [ ] **Il tuo modello hybrid è in training!**

---

## 🎉 Prossimi Step

1. ✅ Completa training hybrid (o riavvia se interrotto)
2. ✅ Deploy: `.\ollama-cli.ps1 deploy --project hybrid_expert --test`
3. ✅ Test UI con selector modelli
4. ✅ Valida risposte F1 e Tolkien
5. ✅ Itera con più dati se necessario

---

**Tutto pronto! Hai un sistema completo e professionale! 🚀**

Usa `.\ollama-cli.ps1 --help` per vedere tutti i comandi disponibili.
