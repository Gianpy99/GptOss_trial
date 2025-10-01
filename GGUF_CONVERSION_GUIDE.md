# 🔄 Guida Conversione GGUF per GPU Stabile

## 🎯 Obiettivo
Convertire il modello fine-tuned in formato GGUF per usarlo con:
- **llama.cpp** (GPU stabile, no transformers)
- **Ollama** (integrazione nativa)
- **LM Studio** (se supporta GGUF Gemma 3)

---

## 📦 Step 1: Installa llama.cpp

### Opzione A: Build da Source (Windows)

```powershell
# 1. Prerequisiti
# - Visual Studio 2022 con C++ tools
# - CMake
# - Git

# 2. Clone repository
cd C:\Development
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 3. Build con CUDA
mkdir build
cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release

# Eseguibile sarà in: build\bin\Release\llama-cli.exe
```

### Opzione B: Download Pre-built (Più Facile)

```powershell
# Download da GitHub Releases
# https://github.com/ggerganov/llama.cpp/releases

# Estrai in: C:\Development\llama.cpp\
```

---

## 🔧 Step 2: Converti Modello in GGUF

### 2.1 - Verifica Modello Merged

```powershell
# Il modello merged deve esistere
dir C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert_merged

# Dovrebbe contenere:
# - model-00001-of-00002.safetensors
# - model-00002-of-00002.safetensors
# - config.json
# - tokenizer.json
```

### 2.2 - Conversione con Script Python

```powershell
# Nel repository llama.cpp
cd C:\Development\llama.cpp

# Attiva ambiente Python
.\.venv_training\Scripts\Activate.ps1

# Installa dipendenze conversione
pip install -r requirements.txt

# CONVERSIONE (può richiedere 5-10 minuti)
python convert_hf_to_gguf.py \
    "C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert_merged" \
    --outfile "C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert.gguf" \
    --outtype f16
```

**Output atteso:**
- File: `f1_expert.gguf` (~8 GB)
- Formato: Float16 (precision originale)

---

## ⚡ Step 3: Quantizza (Opzionale ma Raccomandato)

Quantizzazione riduce dimensione e aumenta velocità:

```powershell
cd C:\Development\llama.cpp\build\bin\Release

# Q4_K_M: Buon bilanciamento qualità/velocità (4-bit)
.\llama-quantize.exe `
    "C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert.gguf" `
    "C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert_q4.gguf" `
    Q4_K_M

# Q5_K_M: Più qualità (5-bit)
.\llama-quantize.exe `
    "C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert.gguf" `
    "C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert_q5.gguf" `
    Q5_K_M
```

**Dimensioni attese:**
- F16: ~8 GB
- Q5_K_M: ~3 GB
- Q4_K_M: ~2.5 GB

---

## 🚀 Step 4: Test con llama.cpp

```powershell
cd C:\Development\llama.cpp\build\bin\Release

# Test inferenza
.\llama-cli.exe `
    -m "C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert_q4.gguf" `
    -n 150 `
    -p "Tell me about Lewis Hamilton's performance in F1." `
    --n-gpu-layers 35

# -n 150: Max 150 token di risposta
# --n-gpu-layers 35: Usa GPU (sperimenta 20-35)
```

**Se funziona**: Vedrai risposta in 1-3 secondi! 🎉

---

## 🎨 Step 5: Integra con Ollama

### 5.1 - Crea Modelfile

```powershell
# In: C:\Development\Ollama_wrapper\
New-Item -ItemType File -Name "Modelfile-gguf" -Force
```

**Contenuto Modelfile-gguf:**
```
FROM ./fine_tuned_models/f1_expert_q4.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"

TEMPLATE """{{ .Prompt }}"""

SYSTEM "You are an F1 expert assistant. Provide detailed information about Formula 1 drivers, teams, circuits, and statistics."
```

### 5.2 - Importa in Ollama

```powershell
cd C:\Development\Ollama_wrapper

# Crea modello in Ollama
ollama create f1-expert-gguf -f Modelfile-gguf

# Verifica
ollama list

# Test
ollama run f1-expert-gguf "Tell me about Lewis Hamilton"
```

---

## 🎮 Step 6: UI Gradio con Ollama

```python
# ui_ollama_gguf.py
import gradio as gr
import requests

OLLAMA_API = "http://localhost:11434/api/generate"

def chat_ollama(message, history):
    response = requests.post(
        OLLAMA_API,
        json={
            "model": "f1-expert-gguf",
            "prompt": message,
            "stream": False
        }
    )
    return response.json()["response"]

demo = gr.ChatInterface(
    chat_ollama,
    title="🏎️ F1 Expert (GGUF + GPU)",
    examples=[
        "Tell me about Lewis Hamilton",
        "McLaren lap times",
        "Monaco GP winner"
    ]
)

demo.launch(server_port=7864)
```

---

## ✅ Checklist Completa

- [ ] llama.cpp clonato e compilato con CUDA
- [ ] Modello merged verificato
- [ ] Conversione GGUF completata (`f1_expert.gguf`)
- [ ] Quantizzazione Q4_K_M eseguita
- [ ] Test con llama-cli.exe riuscito
- [ ] Modelfile creato
- [ ] Modello importato in Ollama
- [ ] Test con `ollama run` funzionante
- [ ] UI Gradio connessa

---

## 🐛 Troubleshooting

### "convert_hf_to_gguf.py not found"
```powershell
cd C:\Development\llama.cpp
git pull  # Aggiorna repo
```

### "Unsupported architecture: gemma3"
Gemma 3 potrebbe non essere supportato ancora in llama.cpp.
Soluzione: Aspetta update o usa Gemma 2.

### "CUDA out of memory" durante conversion
```powershell
# Usa CPU per conversione
python convert_hf_to_gguf.py ... --outtype f16 --use-cpu
```

### Ollama non riconosce GGUF
```powershell
# Verifica path nel Modelfile
FROM ./fine_tuned_models/f1_expert_q4.gguf  # Path relativo
# O
FROM C:/Development/Ollama_wrapper/fine_tuned_models/f1_expert_q4.gguf  # Assoluto
```

---

## 📊 Performance Attese

| Setup | Tempo Risposta | VRAM | Qualità |
|-------|---------------|------|---------|
| llama.cpp Q4_K_M | **1-3 sec** | 2.5 GB | ⭐⭐⭐⭐ |
| llama.cpp Q5_K_M | **2-4 sec** | 3 GB | ⭐⭐⭐⭐⭐ |
| llama.cpp F16 | **3-5 sec** | 8 GB | ⭐⭐⭐⭐⭐ |
| Transformers CPU | 15-20 sec | 0 GB | ⭐⭐⭐⭐⭐ |

**Obiettivo**: **Q4_K_M con GPU = 1-3 secondi!** 🚀

---

## 💡 Vantaggi GGUF

✅ **GPU stabile** (no transformers bugs)  
✅ **Più veloce** (ottimizzazioni llama.cpp)  
✅ **Meno VRAM** (quantizzazione efficiente)  
✅ **Integrazione Ollama** (facile da usare)  
✅ **Portable** (un file .gguf)

---

## 🎯 Prossimo Comando

```powershell
# Inizia con clone llama.cpp
cd C:\Development
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Poi segui build instructions sopra
```

Vuoi che ti aiuti con il primo step? 🚀
