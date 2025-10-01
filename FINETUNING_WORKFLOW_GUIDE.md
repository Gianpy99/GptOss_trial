# 🚀 Workflow Fine-Tuning Professionale

## 📋 Panoramica

Sistema modulare per fine-tuning su **Computer A** e inference su **Computer B**.

### Vantaggi
- ✅ Training su GPU potente, inference su qualsiasi PC
- ✅ Adapter piccoli (~20-50MB) facili da trasferire
- ✅ Riutilizzabile per qualsiasi dataset
- ✅ Usa sempre Ollama per inference (API coerente)

---

## 🔧 Setup Iniziale

### Computer A (Training)
```powershell
# Installa dipendenze fine-tuning
pip install -r requirements-finetuning.txt
```

### Computer B (Inference)
```powershell
# Solo Ollama + wrapper base
pip install -r requirements.txt
# + Ollama installato localmente
```

---

## 📊 FASE 1: Training (Computer A)

### Esempio: Dataset Formula 1

```powershell
python finetuning_workflow.py train `
  --dataset "Vadera007/Formula_1_Dataset" `
  --project "f1_expert" `
  --model "gemma3:4b" `
  --type "f1" `
  --epochs 3 `
  --batch-size 4 `
  --limit 100
```

### Esempio: Dataset Custom QA

```powershell
# Prepara file: my_qa_data.json
# [
#   {"question": "...", "answer": "..."},
#   {"question": "...", "answer": "..."}
# ]

python finetuning_workflow.py train `
  --dataset "my_qa_data.json" `
  --project "my_expert" `
  --model "gemma3:4b" `
  --type "qa" `
  --epochs 3
```

### Esempio: Qualsiasi Dataset Hugging Face

```powershell
python finetuning_workflow.py train `
  --dataset "username/dataset-name" `
  --project "my_project" `
  --model "gemma3:4b" `
  --type "generic" `
  --epochs 5
```

### Output del Training

```
finetuning_projects/
└── f1_expert/
    ├── training_data.json    (dati usati)
    ├── metadata.json          (info progetto)
    └── adapter/
        └── final_adapter/     (adapter LoRA ~20-50MB)
            ├── adapter_config.json
            ├── adapter_model.safetensors
            └── ...
```

---

## 📤 FASE 2: Trasferimento

### Copia la cartella del progetto su Computer B

```powershell
# Computer A → USB/Cloud/Network
Copy-Item -Recurse "finetuning_projects/f1_expert" -Destination "E:/"

# Computer B ← USB/Cloud/Network
Copy-Item -Recurse "E:/f1_expert" -Destination "C:/Development/Ollama_wrapper/finetuning_projects/"
```

**Dimensione**: ~20-100MB (dipende dal modello)

---

## 🚀 FASE 3: Deploy (Computer B)

### Conversione e Import in Ollama

```powershell
python finetuning_workflow.py deploy --project "f1_expert"
```

**⚠️ NOTA**: La conversione GGUF richiede tool aggiuntivi:
- `llama.cpp` per conversione
- `ollama create` per import

### Workflow Manuale (Temporaneo)

Fino all'implementazione automatica:

```powershell
# 1. Converti adapter in GGUF
# TODO: Script di conversione

# 2. Crea Modelfile
echo "FROM gemma3:4b" > Modelfile
echo "ADAPTER ./adapter_gguf" >> Modelfile

# 3. Import in Ollama
ollama create gemma3-4b-f1 -f Modelfile
```

---

## 🧪 FASE 4: Test Comparison (Computer B)

### Confronta Base vs Fine-Tuned

```powershell
python finetuning_workflow.py test --project "f1_expert"
```

### Output Esempio

```
📊 TESTING: BASE vs FINE-TUNED
===============================================

🤖 Base model: gemma3:4b
🎯 Fine-tuned: gemma3-4b-f1

--- Question 1 ---
Q: What team does VER drive for in Formula 1?

[BASE]: VER could refer to various things. In motorsports, 
it might be an abbreviation for a driver's name...

[TUNED]: VER drives for Red Bull Racing in Formula 1.
```

---

## 🎯 Uso con OllamaWrapper

### In Python

```python
from src.ollama_wrapper import OllamaWrapper

# Modello BASE
wrapper_base = OllamaWrapper(model="gemma3:4b")
response_base = wrapper_base.chat("What team does VER drive for?")

# Modello FINE-TUNED
wrapper_tuned = OllamaWrapper(model="gemma3-4b-f1")
response_tuned = wrapper_tuned.chat("What team does VER drive for?")

print("BASE:", response_base["message"]["content"])
print("TUNED:", response_tuned["message"]["content"])
```

---

## 📚 Adattare per Tuoi Dataset

### 1. Dataset Custom (JSON)

```json
[
  {
    "instruction": "Come si fa X?",
    "output": "Per fare X, devi seguire questi passi..."
  },
  {
    "instruction": "Spiegami Y",
    "output": "Y è un concetto che..."
  }
]
```

```powershell
python finetuning_workflow.py train `
  --dataset "mio_dataset.json" `
  --project "mio_esperto" `
  --model "gemma3:4b" `
  --type "generic"
```

### 2. Dataset da Hugging Face

Cerca su https://huggingface.co/datasets

```powershell
python finetuning_workflow.py train `
  --dataset "username/dataset-name" `
  --project "progetto" `
  --model "gemma3:4b"
```

### 3. Personalizza Conversione

Modifica `finetuning_workflow.py`:

```python
def _create_custom_training_data(self, rows):
    """Tua logica personalizzata."""
    training_data = []
    
    for row in rows:
        # Estrai campi specifici del tuo dataset
        question = row.get("my_question_field")
        answer = row.get("my_answer_field")
        
        training_data.append({
            "instruction": question,
            "output": answer
        })
    
    return training_data
```

Poi usa: `--type custom`

---

## ⚙️ Parametri Training

### `--epochs`
- **Default**: 3
- **Più epochs**: Migliore fit, ma rischio overfitting
- **Consigliato**: 3-5 per dataset piccoli, 1-2 per grandi

### `--batch-size`
- **Default**: 4
- **GPU piccola**: 1-2
- **GPU grande**: 8-16
- **Troppo alto**: Out of memory error

### `--limit`
- **Default**: 100 righe
- **Training rapido**: 50-100
- **Training completo**: Rimuovi limit (usa tutto il dataset)

### `--model`
- **gemma3:4b** - Veloce, leggero (consigliato)
- **gemma3:8b** - Più potente, richiede più RAM
- **llama3:8b** - Alternative

---

## 🔥 Esempi Pratici

### 1. Quick Test (5 minuti)
```powershell
python finetuning_workflow.py train `
  --dataset "Vadera007/Formula_1_Dataset" `
  --project "f1_quick" `
  --model "gemma3:4b" `
  --type "f1" `
  --epochs 1 `
  --batch-size 2 `
  --limit 20
```

### 2. Training Completo (30+ minuti)
```powershell
python finetuning_workflow.py train `
  --dataset "Vadera007/Formula_1_Dataset" `
  --project "f1_full" `
  --model "gemma3:8b" `
  --type "f1" `
  --epochs 5 `
  --batch-size 8 `
  --limit 500
```

### 3. Tuo Dataset Personale
```powershell
python finetuning_workflow.py train `
  --dataset "mio_dominio.json" `
  --project "esperto_mio_dominio" `
  --model "gemma3:4b" `
  --type "qa" `
  --epochs 3
```

---

## 🐛 Troubleshooting

### "Out of memory"
```powershell
# Riduci batch size
--batch-size 1

# Oppure usa quantizzazione 4-bit (già attiva di default)
```

### "Dataset not found"
```powershell
# Verifica nome esatto su Hugging Face
# Oppure usa file locale con path assoluto
--dataset "C:/path/to/dataset.json"
```

### "Import slow / stuck"
```powershell
# Normale - PyTorch impiega 30-60s a caricare
# Guarda messaggio: "Loading fine-tuning libraries..."
```

### "Model not found in Ollama"
```powershell
# Dopo deploy, verifica import:
ollama list

# Se non c'è, rifai import manuale
```

---

## 📈 Prossimi Passi

1. ✅ Testare workflow con F1 dataset
2. ⏳ Implementare conversione GGUF automatica
3. ⏳ Integrare `ollama create` automatico
4. ⏳ Aggiungere valutazione metriche (accuracy, perplexity)
5. ⏳ Supporto multi-GPU training

---

## 💡 Tips

- **Cache HF**: Primo download lento, poi veloce (cached)
- **Adapter portabili**: Condividi adapter, non modelli interi
- **Test incrementale**: Parti con 1 epoch e 20 esempi
- **Backup adapter**: Sono piccoli, salvali per esperimenti futuri
- **Naming convention**: `<base-model>-<domain>` (es. `gemma3-4b-f1`)

---

**Domande? Controlla il codice in `finetuning_workflow.py`**
