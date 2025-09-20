# Ollama_wrapper

Lightweight and complete Python wrapper for loca### Streaming chat
```python
# Streaming chat (chunks arrive as they come)
for chunk in wrapper.stream_chat("Write a short poem about Mars"):
    print(chunk, end="", flush=True)
```ma HTTP API (tested with Gemma3 models).

This repository provides a single-file utility, `wrapper.py`, that offers a high-level API for:

- Blocking and streaming chat against a local Ollama server
- Multimodal attachments (images, PDFs) encoded as base64
- Session save/load (JSON under `ollama_sessions/`)
- Memory (SQLite) for conversation history and long-term facts (`ollama_memory.db`)
- Basic model management and safe CLI wrapper for the `ollama` binary

The project is intentionally small and designed for local experimentation with an Ollama server (default base URL: `http://localhost:11434/api`).

## 📋 Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Complete Examples](#complete-examples)
- [Main API](#main-api)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)

## 🔧 Requirements

- Python 3.8+
- `requests` library
- Ollama server running (with `gemma3:4b` model or similar)

## 🚀 Installation

### Quick installation
```powershell
# Clone the repository
git clone https://github.com/Gianpy99/Ollama_wrapper.git
cd Ollama_wrapper

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Installation verification
```powershell
# Run quick test
python test_wrapper.py

# Run complete demo
python demo.py
```

## ⚡ Quick Start

### Simple chat
```python
from ollama_wrapper import OllamaWrapper

# Create wrapper (uses gemma3:4b as default)
wrapper = OllamaWrapper()

# Simple chat
response = wrapper.chat("Explain recursion in simple terms")
print(response['assistant'])
```

### Chat streaming
```python
# Streaming chat (i chunk arrivano man mano)
for chunk in wrapper.stream_chat("Scrivi una breve poesia su Marte"):
    print(chunk, end="", flush=True)
```

### Preconfigured assistants
```python
from ollama_wrapper import create_coding_assistant, create_creative_assistant

# Programming assistant (low temperature, specific prompt)
coding = create_coding_assistant()
code_response = coding.chat("Write a Python function for quicksort")

# Creative assistant (high temperature)
creative = create_creative_assistant()
story = creative.chat("Invent a short story about robots")
```

### Memory and sessions
```python
# Store information
wrapper.store_memory("preferred_language", "Python", "preferences")

# Retrieve information
memory = wrapper.recall_memory("preferred_language")
print(memory)  # ('preferred_language', 'Python', 'preferences')

# Save/load sessions
wrapper.save_session("my_session")
wrapper.load_session("my_session")
```

## 📚 Esempi Completi

### Allegati multimodali
```python
# Chat con allegati (immagini, PDF)
response = wrapper.chat(
    "Analizza questa immagine",
    files=["./immagine.jpg", "./documento.pdf"]
)
```

### Configurazione avanzata
```python
from ollama_wrapper import OllamaWrapper, ModelParameters

# Parametri personalizzati
params = ModelParameters(
    temperature=0.1,        # Creatività bassa per risposte precise
    top_p=0.9,
    max_tokens=2048,
    seed=42                 # Per riproducibilità
)

wrapper = OllamaWrapper(
    model_name="gemma3:4b",
    session_id="sessione_lavoro",
    parameters=params
)

# Imposta un prompt di sistema
wrapper.set_system_prompt("Sei un esperto consulente Python. Rispondi sempre con esempi di codice.")
```

### REPL interattivo
```python
from ollama_wrapper import interactive_repl

wrapper = OllamaWrapper()
interactive_repl(wrapper)  # Avvia REPL interattivo
```

### Gestione modelli
```python
# Lista modelli disponibili
models = wrapper.list_models()
print(models)

# Scarica nuovo modello
result = wrapper.pull_model("gemma3:12b")

# Informazioni su un modello
info = wrapper.show_model_info("gemma3:4b")
```

## 🔧 API Principale

### Classe OllamaWrapper

#### Costruttore
```python
OllamaWrapper(
    base_url="http://localhost:11434/api",
    model_name="gemma3:4b",
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

## 🤖 Fine-tuning e personalizzazione

Il wrapper è progettato per essere facilmente estendibile per progetti di fine-tuning:

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
