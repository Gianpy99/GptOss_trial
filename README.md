# 🏎️ F1 Expert - Fine-tuned Gemma 3 Model# Ollama_wrapper



Fine-tuned **Gemma 3 4B** model specializzato in Formula 1, con **GPU inference** tramite Ollama.Lightweight and complete Python wrapper for loca### Streaming chat

```python

---# Streaming chat (chunks arrive as they come)

for chunk in wrapper.stream_chat("Write a short poem about Mars"):

## 🚀 Quick Start    print(chunk, end="", flush=True)

```ma HTTP API (tested with Gemma3 models).

### Requisiti

- Python 3.11+ con CUDA supportThis repository provides a single-file utility, `wrapper.py`, that offers a high-level API for:

- Ollama installato

- NVIDIA GPU (consigliata: 6GB+ VRAM)- Blocking and streaming chat against a local Ollama server

- Multimodal attachments (images, PDFs) encoded as base64

### Installazione- Session save/load (JSON under `ollama_sessions/`)

- Memory (SQLite) for conversation history and long-term facts (`ollama_memory.db`)

```powershell- Basic model management and safe CLI wrapper for the `ollama` binary

# 1. Clona repository

git clone https://github.com/Gianpy99/Ollama_wrapper.gitThe project is intentionally small and designed for local experimentation with an Ollama server (default base URL: `http://localhost:11434/api`).

cd Ollama_wrapper

## 📋 Table of Contents

# 2. Crea ambiente virtuale- [Requirements](#requirements)

python -m venv .venv_training- [Installation](#installation)

.\.venv_training\Scripts\Activate.ps1- [Quick Start](#quick-start)

- [Complete Examples](#complete-examples)

# 3. Installa dipendenze- [Main API](#main-api)

pip install -r requirements.txt- [Configuration](#configuration)

pip install -r requirements-finetuning.txt- [Testing](#testing)

```- [Contributing](#contributing)



### Uso - Inference con GPU## 🔧 Requirements



```powershell- Python 3.8+

# 1. Importa modello in Ollama- `requests` library

ollama create f1-expert -f Modelfile-safetensors- Ollama server running (with `gemma3:4b` model or similar)



# 2. Avvia UI Gradio## 🚀 Installation

.\.venv_training\Scripts\Activate.ps1

python ui_ollama.py### Quick installation

```powershell

# 3. Apri browser# Clone the repository

# http://localhost:7860git clone https://github.com/Gianpy99/Ollama_wrapper.git

```cd Ollama_wrapper



**Alternative CLI**:# Install dependencies

```powershellpip install -r requirements.txt

# Test diretto con Ollama

ollama run f1-expert "Tell me about Lewis Hamilton"# Install package in editable mode

pip install -e .

# Test Python CPU```

python test_inference_cpu.py

```### Installation verification

```powershell

---# Run quick test

python test_wrapper.py

## 📁 Struttura Progetto

# Run complete demo

```python demo.py

Ollama_wrapper/

├── ui_ollama.py                          # 🎨 UI Gradio + Ollama (GPU)# Test fine-tuning integration (optional)

├── Modelfile-safetensors                 # ⚙️ Config Ollamapython test_integration.py

├── test_inference_cpu.py                 # 🧪 Test CPU alternativo```

├── create_specific_f1_dataset.py         # 📊 Creazione dataset

├── test_training_lightweight.py          # 🏋️ Script training## 🏎️ Formula 1 Fine-Tuning Demo

├── wrapper.py                            # 🔧 Core wrapper

├── f1_training_data.json                 # 📝 Dataset F1 (50 examples)**NEW!** Demo pratica che mostra il miglioramento del modello prima/dopo fine-tuning:

├── requirements.txt                      # 📦 Dependencies base

├── requirements-finetuning.txt           # 📦 Dependencies training```powershell

├── fine_tuned_models/# Demo completa con dataset F1 reale

│   └── f1_expert_merged/                 # 💾 Modello safetensors (8 GB)python demo_f1_finetuning.py

├── finetuning_projects/

│   └── f1_expert_fixed/                  # 📂 Artifacts training# O test veloce (5 minuti)

│       ├── adapter/                      # 🎯 LoRA adapter (45 MB)python test_f1_quick.py

│       ├── checkpoint-*/                 # 💾 Checkpoints```

│       └── runs/                         # 📊 TensorBoard logs

└── ollama_sessions/                      # 💬 Sessioni salvate**Cosa dimostra:**

```- ❌ PRIMA: Modello dà risposte generiche su F1

- ✅ DOPO: Modello è un esperto F1 con risposte accurate

---- 📊 Usa dati reali da Hugging Face

- ⚡ Training in ~10 minuti

## 🎯 Features

Vedi [F1_QUICKSTART.md](F1_QUICKSTART.md) per dettagli completi.

- ✅ **Fine-tuned** su 50 esempi Formula 1

- ✅ **GPU Inference** (1-5 sec per risposta via Ollama)## ⚡ Quick Start

- ✅ **CPU Fallback** (10-15 sec via transformers)

- ✅ **Web UI** (Gradio)### Simple chat

- ✅ **CLI Support** (Ollama + Python)```python

- ✅ **LoRA Adapter** (efficiente, 45 MB)from ollama_wrapper import OllamaWrapper



---# Create wrapper (uses gemma3:4b as default)

wrapper = OllamaWrapper()

## 🏋️ Training

# Simple chat

Il modello è stato fine-tuned su **Formula_1_Dataset** (50 esempi):response = wrapper.chat("Explain recursion in simple terms")

print(response['assistant'])

**Parametri**:```

- Base Model: `google/gemma-3-4b-it`

- LoRA: rank=8, alpha=16### Chat streaming

- Batch size: 2, gradient accumulation: 4```python

- Epochs: 3# Streaming chat (i chunk arrivano man mano)

- Learning rate: 2e-4for chunk in wrapper.stream_chat("Scrivi una breve poesia su Marte"):

- Training time: ~85 minuti (NVIDIA GTX 1660 SUPER)    print(chunk, end="", flush=True)

```

**Risultati**:

- Loss iniziale: 10.99### Preconfigured assistants

- Loss finale: 9.21```python

- Adapter size: 45.5 MBfrom ollama_wrapper import create_coding_assistant, create_creative_assistant

- Merged model: 8 GB

# Programming assistant (low temperature, specific prompt)

---coding = create_coding_assistant()

code_response = coding.chat("Write a Python function for quicksort")

## 📊 Performance

# Creative assistant (high temperature)

| Metodo | Tempo Risposta | VRAM | Device |creative = create_creative_assistant()

|--------|---------------|------|--------|story = creative.chat("Invent a short story about robots")

| **Ollama GPU** | **1-5 sec** | **4-5 GB** | **✅ GPU** |```

| CPU Transformers | 10-15 sec | 0 GB | ✅ CPU |

### Memory and sessions

---```python

# Store information

## 🛠️ Troubleshootingwrapper.store_memory("preferred_language", "Python", "preferences")



### Ollama non risponde# Retrieve information

```powershellmemory = wrapper.recall_memory("preferred_language")

# Verifica che Ollama sia in esecuzioneprint(memory)  # ('preferred_language', 'Python', 'preferences')

ollama list

# Save/load sessions

# Se vuoto, importa modellowrapper.save_session("my_session")

ollama create f1-expert -f Modelfile-safetensorswrapper.load_session("my_session")

``````



### Import Error## 📚 Esempi Completi

```powershell

# Reinstalla dipendenze### Allegati multimodali

pip install --upgrade -r requirements.txt```python

```# Chat con allegati (immagini, PDF)

response = wrapper.chat(

### GPU non utilizzata    "Analizza questa immagine",

Ollama gestisce GPU automaticamente. Verifica con:    files=["./immagine.jpg", "./documento.pdf"]

```powershell)

nvidia-smi  # Durante inferenza dovresti vedere Ollama usando VRAM```

```

### Configurazione avanzata

---```python

from ollama_wrapper import OllamaWrapper, ModelParameters

## 📚 Guide

# Parametri personalizzati

- `SOLUTION_FINAL_OLLAMA.md` - Soluzione completa GPUparams = ModelParameters(

- `FINETUNING_GUIDE.md` - Guida training completa    temperature=0.1,        # Creatività bassa per risposte precise

- `FINETUNING_SUMMARY.md` - Summary training    top_p=0.9,

- `QUICK_REFERENCE.md` - Comandi rapidi    max_tokens=2048,

- `TROUBLESHOOTING.md` - Problemi comuni    seed=42                 # Per riproducibilità

)

---

wrapper = OllamaWrapper(

## 🤝 Contributing    model_name="gemma3:4b",

    session_id="sessione_lavoro",

Pull requests welcome! Per modifiche importanti, apri prima una issue.    parameters=params

)

---

# Imposta un prompt di sistema

## 📄 Licensewrapper.set_system_prompt("Sei un esperto consulente Python. Rispondi sempre con esempi di codice.")

```

MIT License - vedi LICENSE file

### REPL interattivo

---```python

from ollama_wrapper import interactive_repl

## 👤 Author

wrapper = OllamaWrapper()

**Gianpy99**interactive_repl(wrapper)  # Avvia REPL interattivo

- GitHub: [@Gianpy99](https://github.com/Gianpy99)```



---### Gestione modelli

```python

## 🙏 Credits# Lista modelli disponibili

models = wrapper.list_models()

- **Google Gemma 3** - Base modelprint(models)

- **HuggingFace** - Transformers & PEFT

- **Ollama** - GPU inference backend# Scarica nuovo modello

- **Gradio** - Web UI frameworkresult = wrapper.pull_model("gemma3:12b")



---# Informazioni su un modello

info = wrapper.show_model_info("gemma3:4b")

## 📈 Stats```



![Model](https://img.shields.io/badge/Model-Gemma_3_4B-blue)## 🔧 API Principale

![Status](https://img.shields.io/badge/Status-Production-green)

![GPU](https://img.shields.io/badge/GPU-Supported-brightgreen)### Classe OllamaWrapper

![License](https://img.shields.io/badge/License-MIT-yellow)

#### Costruttore

---```python

OllamaWrapper(

**Last Updated**: 01/10/2025      base_url="http://localhost:11434/api",

**Version**: 1.0.0    model_name="gemma3:4b",

    session_id="default",
    parameters=None,
    memory_db_path="ollama_memory.db",
    prefer_cli=False
)
```

#### Metodi principali
- `chat(message, include_history=True, store_conversation=True, files=None, timeout=60)` - Chat bloccante
- `stream_chat(message, include_history=True, timeout=120)` - Chat streaming
- `set_system_prompt(prompt)` - Imposta prompt di sistema
- `save_session(name)` / `load_session(name)` - Gestione sessioni
- `store_memory(key, value, category="general")` - Memorizza fatto
- `recall_memory(key)` - Recupera fatto
- `search_memories(query, limit=20)` - Cerca nella memoria

### Assistenti preconfigurati
- `create_coding_assistant(session_id="coding")` - Per programmazione
- `create_creative_assistant(session_id="creative")` - Per contenuti creativi

### Classe MemoryManager
- `store_message(session_id, message)` - Memorizza messaggio conversazione
- `get_conversation_history(session_id, limit=100)` - Recupera cronologia
- `store_fact(key, value, category)` - Memorizza fatto
- `search_facts(query, limit=20)` - Cerca fatti

## ⚙️ Configurazione

### URL base e modello predefinito
```python
DEFAULT_BASE_URL = "http://localhost:11434/api"
DEFAULT_MODEL = "gemma3:4b"
```

### Directory di output
- `ollama_sessions/` - File di sessione (JSON)
- `ollama_memory.db` - Database SQLite per memoria

### Parametri del modello
```python
ModelParameters(
    temperature=0.7,      # Creatività (0.0-1.0)
    top_p=0.9,           # Nucleus sampling
    top_k=None,          # Top-k sampling
    max_tokens=1024,     # Lunghezza massima risposta
    repeat_penalty=None, # Penalità ripetizione
    seed=None,           # Seed per riproducibilità
    num_ctx=None         # Contesto (lunghezza)
)
```

## 🧪 Test

### Test rapido
```powershell
python test_wrapper.py
```

### Test completi
```powershell
python test_complete.py
```

### Demo interattiva
```powershell
python demo.py
```

### Test unitari
```powershell
python -m pytest tests/
```

## 🤖 Fine-Tuning with Hugging Face & PEFT

**NEW!** Ollama_wrapper ora supporta il fine-tuning di modelli usando le tue conversazioni!

### Installazione dipendenze fine-tuning
```powershell
# Installa dipendenze per fine-tuning
pip install -r requirements-finetuning.txt

# Oppure usa le dipendenze opzionali
pip install -e .[finetuning]
```

### Quick Start Fine-Tuning
```python
from ollama_wrapper import OllamaWrapper, FineTuningManager

# 1. Crea conversazioni di training con Ollama
wrapper = OllamaWrapper(session_id="training")
wrapper.chat("What are Python decorators?")
wrapper.chat("Show me an example")

# 2. Fine-tune un modello usando quelle conversazioni
manager = FineTuningManager(
    model_name="microsoft/phi-2",
    use_4bit=True  # QLoRA per efficienza memoria
)

manager.load_model()
manager.setup_lora(r=16, lora_alpha=32)

# 3. Carica dati dal database memoria di Ollama
dataset = manager.load_training_data_from_memory()
tokenized = manager.tokenize_dataset(dataset)

# 4. Addestra
manager.train(tokenized, num_epochs=3, output_name="my_assistant")
manager.save_adapter("my_assistant_adapter")  # Solo pochi MB!

# 5. Usa il modello fine-tuned
from ollama_wrapper import create_finetuned_assistant
assistant = create_finetuned_assistant("./fine_tuned_models/my_assistant_adapter")
response = assistant.generate_text("User: Explain decorators\n\nAssistant:")
```

### Caratteristiche Fine-Tuning
- ✅ **Integrazione seamless** con la memoria di OllamaWrapper
- ✅ **PEFT/LoRA** per training efficiente (solo pochi MB di adapter)
- ✅ **QLoRA** con quantizzazione 4-bit per basso uso memoria
- ✅ **Multiple data sources**: memoria SQLite, JSON, formati custom
- ✅ **Adapters multipli**: crea assistenti specializzati per task diversi
- ✅ **Workflow ibrido**: combina Ollama (veloce) con modelli fine-tuned (specializzati)

### Esempi Completi
```powershell
# Quick start (5 minuti)
python examples/quick_start_finetuning.py

# Workflow completo interattivo
python examples/example_finetuning_integration.py

# Esegui step singoli
python examples/example_finetuning_integration.py step1  # Crea dati
python examples/example_finetuning_integration.py step2  # Fine-tune
python examples/example_finetuning_integration.py step3  # Testa
```

### Documentazione Completa
Vedi [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) per:
- Guida dettagliata setup
- Best practices e troubleshooting
- Configurazioni LoRA avanzate
- Workflow ibridi Ollama + fine-tuned
- Esempi di integrazione

### Preparazione dati per fine-tuning
```python
# Esporta conversazioni per training
wrapper = OllamaWrapper(session_id="training_data")
history = wrapper.memory.get_conversation_history("training_data", limit=1000)

# Formatta per fine-tuning
training_data = []
for msg in history:
    training_data.append({
        "role": msg.role,
        "content": msg.content,
        "timestamp": msg.timestamp
    })
```

### Integrazione con altri servizi
```python
# Esempio: integrazione con logging avanzato
import logging

class ExtendedWrapper(OllamaWrapper):
    def chat(self, message, **kwargs):
        logging.info(f"Chat request: {message[:50]}...")
        response = super().chat(message, **kwargs)
        logging.info(f"Chat response: {response.get('status')}")
        return response
```

## 🛠️ Sviluppo e Contributi

### Note per sviluppatori
- Evita chiamate di rete reali nei test unitari; mocka `requests.get` e `requests.post`
- Test unitari utili da aggiungere:
  - Creazione schema `MemoryManager` e operazioni CRUD (usa file DB temporaneo)
  - `_build_messages()` con e senza `system_prompt` e con cronologia memorizzata
  - Gestione `stream_chat()` di stream JSON-line e chunk di testo (mocka `requests.post(..., stream=True)`)

### Casi limite e gotchas
- Gli endpoint streaming possono emettere linee non-JSON — `stream_chat` fallback su chunk raw
- `pull_model()` aspetta linee di eventi streaming da `/pull` e ritorna eventi parsed (se `stream=True`) o body completo
- Le connessioni SQLite sono create con `check_same_thread=False` e ogni operazione apre una nuova connessione

### Struttura del progetto
```
Ollama_wrapper/
├── src/ollama_wrapper/          # Package principale
│   ├── __init__.py              # Export pubblici
│   └── wrapper.py               # Implementazione principale
├── examples/                    # Esempi di utilizzo
├── tests/                       # Test unitari e integrazione
├── scripts/                     # Script di utilità
├── demo.py                      # Demo interattiva
├── test_wrapper.py             # Test rapido
├── test_complete.py            # Test completi
├── requirements.txt            # Dipendenze
├── pyproject.toml             # Configurazione package
└── README.md                  # Questa documentazione
```

### Contribuire
- Mantieni le modifiche localizzate su `wrapper.py` quando possibile
- Aggiungi test sotto la directory `tests/`
- Se aggiungi nuove dipendenze, includile in `requirements.txt`

## 🐛 Risoluzione problemi

### Problemi comuni

**Errore: "Connection refused"**
```bash
# Verifica che Ollama sia in esecuzione
curl http://localhost:11434/api/tags

# Avvia Ollama se non è attivo
ollama serve
```

**Errore: "Model not found"**
```bash
# Lista modelli disponibili
ollama list

# Scarica il modello se necessario
ollama pull gemma3:4b
```

**Timeout su chat lunghe**
```python
# Aumenta il timeout
response = wrapper.chat("Domanda complessa", timeout=120)
```

**Problemi con memoria SQLite**
```python
# Resetta il database se corrotto
import os
os.remove("ollama_memory.db")
wrapper = OllamaWrapper()  # Ricreerà il DB
```

## 📄 Licenza

MIT License - vedi il file LICENSE per dettagli.

## 🤝 Supporto

- Apri un issue per bug o richieste di funzionalità
- Controlla la documentazione in `.github/copilot-instructions.md` per dettagli implementativi
- Esegui `python demo.py` per vedere esempi di utilizzo

---

**🚀 Buon coding con Ollama!**
