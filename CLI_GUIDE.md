# 🏎️ Ollama Wrapper CLI - Quick Start

## 🚀 Comandi Rapidi

### 📋 Lista Risorse

```powershell
# Lista tutto
.\ollama-cli.ps1 list

# Solo modelli Ollama
.\ollama-cli.ps1 list --type models

# Solo dataset disponibili
.\ollama-cli.ps1 list --type datasets

# Solo progetti di training
.\ollama-cli.ps1 list --type projects
```

### 🏋️ Training

```powershell
# Training base (50 esempi, 2 epochs)
.\ollama-cli.ps1 train --dataset combined_f1_tolkien_data.json

# Training custom
.\ollama-cli.ps1 train --dataset my_data.json --project my_model --limit 100 --epochs 3

# Training senza auto-merge
.\ollama-cli.ps1 train --dataset my_data.json --no-merge

# Training senza auto-deploy
.\ollama-cli.ps1 train --dataset my_data.json --no-deploy
```

### 🔄 Merge Adapter

```powershell
# Merge progetto esistente
.\ollama-cli.ps1 merge --project hybrid_expert

# Merge senza auto-deploy
.\ollama-cli.ps1 merge --project my_project --no-deploy
```

### 📦 Deploy su Ollama

```powershell
# Deploy con nome custom
.\ollama-cli.ps1 deploy --project hybrid_expert --name my-awesome-model

# Deploy con test automatico
.\ollama-cli.ps1 deploy --project hybrid_expert --test
```

### 🌐 Launch Web UI

```powershell
# UI con selector modelli (CONSIGLIATO)
.\ollama-cli.ps1 ui

# UI con modello specifico
.\ollama-cli.ps1 ui --model hybrid-expert

# UI su porta custom
.\ollama-cli.ps1 ui --port 8080
```

### 🚀 Pipeline Completa

```powershell
# Train + Merge + Deploy + UI in un comando!
.\ollama-cli.ps1 pipeline --dataset combined_f1_tolkien_data.json

# Con parametri custom
.\ollama-cli.ps1 pipeline --dataset my_data.json --project awesome_model --limit 100 --epochs 3
```

---

## 📊 Workflow Tipico

### Scenario 1: Nuovo Modello da Zero

```powershell
# 1. Prepara dataset (se necessario)
python combine_datasets.py

# 2. Pipeline completa
.\ollama-cli.ps1 pipeline --dataset combined_f1_tolkien_data.json --project my_model

# Questo eseguirà automaticamente:
#   ✅ Training
#   ✅ Merge adapter
#   ✅ Deploy su Ollama
#   ✅ Launch Web UI
```

### Scenario 2: Re-deploy Modello Esistente

```powershell
# 1. Lista progetti
.\ollama-cli.ps1 list --type projects

# 2. Deploy progetto
.\ollama-cli.ps1 deploy --project hybrid_expert --test

# 3. Launch UI
.\ollama-cli.ps1 ui --model hybrid-expert
```

### Scenario 3: Merge e Deploy Manuale

```powershell
# 1. Training solo
.\ollama-cli.ps1 train --dataset my_data.json --no-merge --no-deploy

# 2. Merge quando pronto
.\ollama-cli.ps1 merge --project my_data --no-deploy

# 3. Deploy quando pronto
.\ollama-cli.ps1 deploy --project my_data --name my-model --test

# 4. UI
.\ollama-cli.ps1 ui --model my-model
```

---

## 🎨 Web UI Features

La nuova UI multi-modello include:

✅ **Selector Modelli Dinamico** - Cambia modello on-the-fly  
✅ **Model Info** - Visualizza parametri del modello corrente  
✅ **Temperature Control** - Regola creatività delle risposte  
✅ **Token Limit** - Controlla lunghezza risposte  
✅ **Example Buttons** - Test rapidi F1/Tolkien  
✅ **Clear Chat** - Reset conversazione  
✅ **Copy Button** - Copia risposte facilmente  

---

## 🔧 Troubleshooting

### Errore: "Ollama non trovato"

```powershell
# Verifica che Ollama sia in esecuzione
ollama list

# Se non parte, avvia manualmente
ollama serve
```

### Errore: "Virtual environment non trovato"

```powershell
# Crea il venv se manca
python -m venv .venv_training

# Installa dependencies
.\.venv_training\Scripts\activate
pip install -r requirements-finetuning.txt
```

### Errore: "Dataset non trovato"

```powershell
# Lista dataset disponibili
.\ollama-cli.ps1 list --type datasets

# Crea dataset combinato
python combine_datasets.py
```

### UI non si carica

```powershell
# Verifica che gradio sia installato
pip install gradio requests

# Prova porta alternativa
.\ollama-cli.ps1 ui --port 8080
```

---

## 📚 Comandi Dettagliati

### `train` - Training Modello

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `--dataset` | File JSON dataset (required) | - |
| `--project` | Nome progetto | Derivato da dataset |
| `--limit` | Numero esempi training | 50 |
| `--epochs` | Numero epoche | 2 |
| `--no-merge` | Skip auto-merge | False |
| `--no-deploy` | Skip auto-deploy | False |

### `merge` - Merge Adapter

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `--project` | Nome progetto (required) | - |
| `--no-deploy` | Skip auto-deploy | False |

### `deploy` - Deploy su Ollama

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `--project` | Nome progetto (required) | - |
| `--name` | Nome in Ollama | Nome progetto |
| `--test` | Test dopo deploy | False |

### `list` - Lista Risorse

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `--type` | Tipo: all/models/datasets/projects | all |

### `ui` - Web UI

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `--model` | Modello Ollama da usare | Primo disponibile |
| `--port` | Porta web server | 7860 |

### `pipeline` - Pipeline Completa

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `--dataset` | File JSON dataset (required) | - |
| `--project` | Nome progetto | Derivato da dataset |
| `--limit` | Numero esempi | 50 |
| `--epochs` | Numero epoche | 2 |

---

## 💡 Tips & Tricks

### Alias PowerShell

Aggiungi al tuo `$PROFILE`:

```powershell
function ollama-cli { & "C:\Development\Ollama_wrapper\ollama-cli.ps1" $args }
```

Ora puoi usare:

```powershell
ollama-cli list
ollama-cli ui
```

### Training Rapido

Per test veloci:

```powershell
.\ollama-cli.ps1 train --dataset my_data.json --limit 20 --epochs 1
```

### Backup Modelli

```powershell
# Esporta modello da Ollama
ollama show my-model --modelfile > my-model.Modelfile

# Lista progetti con status
.\ollama-cli.ps1 list --type projects
```

---

## 🎯 Next Steps

Dopo aver completato il workflow:

1. ✅ Testa il modello nella UI
2. ✅ Valida le risposte
3. ✅ Itera con più dati se necessario
4. ✅ Deploy in produzione

**Enjoy! 🚀**
