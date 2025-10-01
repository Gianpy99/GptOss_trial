# Problemi Comuni e Soluzioni - Fine-Tuning

## 🐛 Errore: "The model did not return a loss from the inputs"

### **Sintomo**
```
ValueError: The model did not return a loss from the inputs, only the following keys: logits. 
For reference, the inputs it received are input_ids,attention_mask.
```

### **Causa**
Il dataset tokenizzato non contiene le `labels` necessarie per il training. 
Per il Causal Language Modeling, il modello calcola la loss confrontando le predizioni con le labels.

### **Soluzione**
Nella funzione di tokenizzazione, aggiungere le labels che sono copie degli input_ids:

```python
def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    # Per Causal LM, labels = input_ids
    result["labels"] = result["input_ids"].copy()
    return result
```

---

## 🐛 Errore: "column names don't match" (Dataset F1)

### **Sintomo**
```
datasets.exceptions.DatasetGenerationCastError: An error occurred while generating the dataset
All the data files must have the same columns, but at some point there are 3 new columns 
({'Year', 'GrandPrix', 'EventDate'})
```

### **Causa**
Il dataset `Vadera007/Formula_1_Dataset` contiene più file CSV con colonne diverse:
- `f1_data_2023_Bahrain.csv` - 9 colonne base
- `f1_historical_data.csv` - 12 colonne (include Year, GrandPrix, EventDate)
- `f1_historical_data_with_features.csv` - 12 colonne

### **Soluzione**
Specificare esplicitamente quale file CSV usare:

```python
# Usa il file più completo
dataset = load_dataset(
    "Vadera007/Formula_1_Dataset", 
    data_files="f1_historical_data.csv", 
    split="train"
)
```

Oppure usa tutti i file ma specifica le colonne comuni:

```python
dataset = load_dataset("Vadera007/Formula_1_Dataset", split="train")
# Poi filtra le colonne
common_cols = ['Driver', 'Team', 'AvgLapTime', 'LapsCompleted', 
               'AirTemp', 'TrackTemp', 'Rainfall', 
               'QualiPosition', 'RaceFinishPosition']
dataset = dataset.select_columns(common_cols)
```

---

## 🐛 ModuleNotFoundError: No module named 'torch'

### **Sintomo**
```
ModuleNotFoundError: No module named 'torch'
```

### **Causa**
Ambiente virtuale non attivato o librerie non installate.

### **Soluzione**

**Per Training (GPU)**:
```powershell
.\.venv_training\Scripts\Activate.ps1
python test_gpu_quick.py
```

**Per Inference**:
```powershell
.\.venv_inference\Scripts\Activate.ps1
python finetuning_workflow.py test --project "f1_expert"
```

Se le librerie non sono installate:
```powershell
.\.venv_training\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft datasets accelerate bitsandbytes
```

---

## 🐛 CUDA Out of Memory

### **Sintomo**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

### **Causa**
Il modello + batch size richiedono più VRAM di quella disponibile.

### **Soluzione**

1. **Riduci batch_size**:
```python
per_device_train_batch_size=1  # invece di 2 o 4
```

2. **Aumenta gradient_accumulation_steps**:
```python
gradient_accumulation_steps=8  # invece di 4
```

3. **Usa 4-bit quantization** (se non già attivo):
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

4. **Per GTX 1660 SUPER (6GB)**:
```python
# Configurazione ottimale
batch_size = 1 o 2
gradient_accumulation_steps = 4 o 8
max_length = 512  # invece di 1024
```

---

## 🐛 Import Lento (30-60 secondi)

### **Sintomo**
Lo script sembra bloccato dopo l'avvio, senza output per 30-60 secondi.

### **Causa**
La prima importazione di `transformers`, `torch` e `bitsandbytes` richiede tempo per:
- Inizializzare CUDA
- Caricare moduli C++/CUDA
- Controllare compatibilità hardware

### **Soluzione**
È **normale** - non è un errore! Import successivi saranno più veloci.

Se vuoi vedere il progresso:
```python
print("Caricamento librerie...")
import torch
print("✓ PyTorch caricato")
from transformers import AutoTokenizer
print("✓ Transformers caricato")
from peft import LoraConfig
print("✓ PEFT caricato")
```

---

## 🐛 UnicodeEncodeError (Emoji su Windows)

### **Sintomo**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3ce' in position X
```

### **Causa**
Windows PowerShell usa codifica cp1252 che non supporta emoji.

### **Soluzione**

Aggiungi all'inizio dello script:
```python
import sys
import io

# Forza UTF-8 su Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

---

## 🐛 HuggingFace Token Non Trovato

### **Sintomo**
```
OSError: You are trying to access a gated repo. Make sure to request access at...
```

### **Causa**
Token HuggingFace non configurato o non valido per il modello gated (Gemma).

### **Soluzione**

1. **Crea file `.env`**:
```bash
HF_TOKEN=........  # Il tuo token HuggingFace
```

2. **Carica nel codice**:
```python
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
```

3. **Usa con HuggingFace**:
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
```

4. **Richiedi accesso al modello**:
   - Vai su https://huggingface.co/google/gemma-3-4b-it
   - Clicca "Request access"
   - Aspetta approvazione (solitamente immediata)

---

## 🐛 Python 3.13 + PyTorch CUDA Incompatibile

### **Sintomo**
```
ERROR: Could not find a version that satisfies the requirement torch
```

### **Causa**
PyTorch non supporta ancora Python 3.13 con CUDA.

### **Soluzione**
Usa la **Dual Environment Strategy**:

```powershell
# Setup automatico
.\scripts\setup_dual_environments_simple.ps1

# Training con Python 3.11
.\.venv_training\Scripts\Activate.ps1

# Inference con Python 3.13
.\.venv_inference\Scripts\Activate.ps1
```

Vedi: `DUAL_ENVIRONMENT_SETUP.md`

---

## 🐛 GPU Non Rilevata (CUDA Available: False)

### **Sintomo**
```python
import torch
print(torch.cuda.is_available())  # False
```

### **Causa**
- PyTorch CPU-only installato
- Python version incompatibile (3.13)
- Driver NVIDIA vecchio

### **Soluzione**

1. **Verifica driver NVIDIA**:
```powershell
nvidia-smi
```

2. **Reinstalla PyTorch con CUDA**:
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Usa Python 3.11** (non 3.13):
```powershell
py -3.11 -m venv .venv_training
.\.venv_training\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 🐛 Warning: "use_cache=True is incompatible with gradient checkpointing"

### **Sintomo**
```
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
```

### **Causa**
Il gradient checkpointing (per risparmiare memoria) disabilita automaticamente la cache del modello.

### **Soluzione**
È un **warning normale** - non è un errore! Il training continuerà correttamente.

Puoi disabilitare esplicitamente:
```python
model.config.use_cache = False
```

---

## 📚 Documentazione Completa

- `DUAL_ENVIRONMENT_SETUP.md` - Setup dual environment
- `GPU_SETUP_FIX.md` - Risoluzione problemi GPU
- `QUICK_REFERENCE.md` - Comandi rapidi
- `FINETUNING_WORKFLOW_GUIDE.md` - Workflow completo

---

**Ultimo aggiornamento**: 1 ottobre 2025
