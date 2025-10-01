# Come Usare il Modello Fine-tuned con LM Studio

## 🎯 LM Studio: La Soluzione PIÙ FACILE (Raccomandato!)

LM Studio può caricare direttamente i modelli HuggingFace senza conversione GGUF!

### Passaggi:

#### 1. **Crea il Modello Merged**
```bash
# Attiva environment
.\.venv_training\Scripts\Activate.ps1

# Esegui merge
python merge_adapter.py
```

Questo crea: `fine_tuned_models/f1_expert_merged/` (~8 GB)

#### 2. **Apri LM Studio**
- Scarica da: https://lmstudio.ai/
- Installa e apri

#### 3. **Importa il Modello**
- `File` → `Import` → `Import from local folder`
- Seleziona: `C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert_merged`
- LM Studio lo carica automaticamente!

#### 4. **Usa il Modello**
- Vai su `Chat`
- Seleziona `f1_expert_merged` dalla lista
- Prova: "Tell me about Lewis Hamilton's performance in F1"

---

## 📊 Confronto Metodi

| Metodo | Difficoltà | Tempo | Pro | Contro |
|--------|-----------|-------|-----|--------|
| **LM Studio** | 🟢 Facile | 10 min | Supporta safetensors, UI grafica | Richiede installazione |
| **Ollama** | 🔴 Difficile | 30-60 min | Integrato CLI | Serve conversione GGUF |
| **Python diretto** | 🟡 Medio | 5 min | Già funziona | Nessuna UI |

---

## 🔧 Alternative per Ollama

Se vuoi **davvero** usare Ollama, ecco le opzioni:

### Opzione A: Conversione GGUF con llama-cpp-python
```bash
# In .venv_training
pip install llama-cpp-python

# Merge
python merge_adapter.py

# Converti (ATTENZIONE: può non funzionare su Windows)
python -m llama_cpp.convert "./fine_tuned_models/f1_expert_merged" \
       --outfile "./fine_tuned_models/f1_expert.gguf" \
       --outtype f16
```

### Opzione B: Usa HuggingFace Spaces
1. Merge: `python merge_adapter.py`
2. Upload su HuggingFace (se hai account)
3. Usa: https://huggingface.co/spaces/ggml-org/gguf-my-repo
4. Scarica il GGUF generato
5. Importa in Ollama con Modelfile

### Opzione C: Wrapper Python + Ollama per Interfaccia
Usa il tuo `wrapper.py` esistente ma carica il modello merged:

```python
# Modifica wrapper.py per usare il merged model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "./fine_tuned_models/f1_expert_merged"
)
# Poi usa l'interfaccia di wrapper.py
```

---

## 📁 Dove Sono i File

```
📂 C:\Development\Ollama_wrapper\
   📂 finetuning_projects\
      📂 f1_expert_fixed\
         📂 adapter\                    ← Adapter LoRA (solo differenze)
            📄 adapter_model.safetensors  (45.5 MB)
            📄 adapter_config.json
            
   📂 fine_tuned_models\               ← Dopo merge_adapter.py
      📂 f1_expert_merged\             ← Modello COMPLETO per LM Studio
         📄 model.safetensors          (~8 GB)
         📄 config.json
         📄 tokenizer files...
         
      📄 f1_expert.gguf                ← Dopo conversione (per Ollama)
         (~4-6 GB, quantizzato)
```

---

## 🎯 RACCOMANDAZIONE FINALE

**Per te:**
1. ✅ **Esegui `merge_adapter.py`** (crea modello completo)
2. ✅ **Scarica LM Studio** (gratis, facile)
3. ✅ **Importa `f1_expert_merged`** in LM Studio
4. ✅ **Testa e confronta** con Ollama `gemma3:4b`

**Vantaggi LM Studio:**
- ✅ Carica safetensors direttamente (no GGUF necessario)
- ✅ UI grafica comoda
- ✅ Supporta quantizzazione automatica (4-bit, 8-bit)
- ✅ Server locale compatibile con API OpenAI
- ✅ Funziona su Windows senza problemi

**Se davvero vuoi Ollama:**
- Serve build llama.cpp su Windows (complesso)
- O usa servizio online per conversione GGUF
- O aspetta che Ollama supporti PEFT nativamente

Vuoi che esegua `merge_adapter.py` per te?
