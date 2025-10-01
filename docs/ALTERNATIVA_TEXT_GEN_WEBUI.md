# 🚀 Alternativa: Text Generation WebUI (oobabooga)

## ❌ Problema con LM Studio

LM Studio potrebbe non supportare:
1. **Gemma 3** (architettura troppo recente)
2. **Safetensors di modelli grandi** senza GGUF

## ✅ Soluzione: Text Generation WebUI

Supporta **QUALSIASI** modello HuggingFace direttamente!

### 🔧 Setup Rapido

```bash
# 1. Clona repository
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui

# 2. Setup automatico Windows
start_windows.bat

# 3. Download dipendenze (automatico al primo avvio)
# Scegli: NVIDIA GPU quando richiesto

# 4. Avvia WebUI
start_windows.bat
```

### 📂 Carica il Tuo Modello

Una volta avviato:

1. Apri browser: http://localhost:7860
2. Vai su **Model** tab
3. Clicca **"Load custom model"**
4. Inserisci path: `C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert_merged`
5. Clicca **Load**
6. Vai su **Chat** tab
7. Inizia a chattare!

### ✅ Vantaggi

- ✅ Supporta **tutti** i modelli HuggingFace
- ✅ Carica safetensors direttamente (no GGUF)
- ✅ UI grafica completa
- ✅ Extensions (memorizzazione, RAG, etc)
- ✅ API compatibile OpenAI
- ✅ Supporto GPU/CPU
- ✅ Quantizzazione automatica

---

## 🎯 Confronto Opzioni

| Opzione | Difficoltà | Supporto Gemma 3 | Tempo Setup |
|---------|-----------|------------------|-------------|
| **Python diretto** | 🟢 Facile | ✅ Funziona | ✅ Già pronto |
| **Text-Gen WebUI** | 🟡 Medio | ✅ Funziona | 🟡 15 min |
| **LM Studio** | 🟢 Facile | ❓ Forse no | ✅ 5 min |
| **Ollama (GGUF)** | 🔴 Difficile | ❓ Serve build | 🔴 30-60 min |

---

## 📋 Passo-Passo Text Generation WebUI

### Windows (Metodo Automatico)

```powershell
# 1. Download one-click installer
# https://github.com/oobabooga/text-generation-webui/releases
# Scarica: oobabooga_windows.zip

# 2. Estrai e apri la cartella

# 3. Esegui
.\start_windows.bat

# 4. Al primo avvio:
#    - Scegli: A (NVIDIA GPU)
#    - Lascia installare tutto (5-10 min)

# 5. Browser si apre su localhost:7860

# 6. Model tab → Load custom model
#    Path: C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert_merged

# 7. Chat tab → Inizia!
```

### Configurazione Avanzata

Se hai problemi di memoria GPU (6GB):

```python
# Nel launcher:
# Modifica start_windows.bat o usa queste opzioni nell'interfaccia:

--load-in-4bit           # Quantizzazione 4-bit (come nel training)
--compute_dtype bfloat16 # Tipo compute
--use-flash-attention-2  # Attenzione ottimizzata
```

---

## 🎯 RACCOMANDAZIONE FINALE

**Per te ora:**

1. ✅ **Continua con Python** (`test_inference_cpu.py`) - **funziona già**
   
2. 🚀 **Installa Text-Gen WebUI** - **migliore per UI grafica**
   - Supporto garantito Gemma 3
   - Carica il tuo merged model direttamente
   - 15 minuti setup
   
3. ⏸️ **Aspetta LM Studio** - potrebbero aggiungere Gemma 3 presto
   
4. ❌ **Evita GGUF/Ollama** - troppo complesso per Gemma 3 ora

---

## 🔗 Link Utili

- **Text-Gen WebUI**: https://github.com/oobabooga/text-generation-webui
- **One-Click Installer**: https://github.com/oobabooga/text-generation-webui/releases
- **Documentazione**: https://github.com/oobabooga/text-generation-webui/wiki

---

Vuoi che ti aiuti con:
- **A)** Setup Text Generation WebUI
- **B)** Migliorare lo script Python esistente (aggiungere UI)
- **C)** Tentare conversione GGUF (se proprio vuoi LM Studio/Ollama)
