# 🚀 Test Rapido llama.cpp CLI

## ✅ Dopo Compilazione

### 1. Trova l'eseguibile
```powershell
dir C:\Development\llama.cpp\build\bin\Release\llama-cli.exe
```

### 2. Test con GGUF
```powershell
cd C:\Development\llama.cpp\build\bin\Release

# Test base (CPU)
.\llama-cli.exe `
    -m "C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert.gguf" `
    -n 150 `
    -p "Tell me about Lewis Hamilton in Formula 1." `
    --threads 8

# Se viene supportata GPU (experimental)
.\llama-cli.exe `
    -m "C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert.gguf" `
    -n 150 `
    -p "Tell me about Lewis Hamilton in Formula 1." `
    --n-gpu-layers 20 `
    --threads 8
```

### 3. Parametri Utili

| Parametro | Descrizione | Esempio |
|-----------|-------------|---------|
| `-m` | Path modello GGUF | `f1_expert.gguf` |
| `-n` | Max tokens risposta | `150` |
| `-p` | Prompt | `"Tell me..."` |
| `--threads` | CPU threads | `8` |
| `--n-gpu-layers` | Layer su GPU | `20` (0=CPU only) |
| `--temp` | Temperature | `0.7` |
| `--top-p` | Top-P sampling | `0.9` |
| `-c` | Context size | `2048` |

### 4. Performance Attese

**CPU only (compilato)**:
- Velocità: ~50-80 tokens/sec
- Tempo risposta: 5-10 secondi
- VRAM: 0 GB

**Confronto con ui_gradio.py**:
- llama.cpp CPU: 5-10 sec ⚡
- ui_gradio.py CPU: 15-20 sec 🐢

---

## 🎯 Comando Finale

```powershell
# Navigate
cd C:\Development\llama.cpp\build\bin\Release

# Run
.\llama-cli.exe -m "C:\Development\Ollama_wrapper\fine_tuned_models\f1_expert.gguf" -n 150 -p "Tell me about Lewis Hamilton"
```

Dovrebbe rispondere in **5-10 secondi** (più veloce di ui_gradio.py)! 🏎️

---

## 📊 Se Funziona

**Opzioni**:
1. ✅ Usa llama-cli.exe direttamente (command line)
2. ✅ Crea script PowerShell wrapper
3. ✅ Integra in ui_llama_cpp.py con `subprocess`

---

## 🐛 Se Non Supporta Gemma 3

Errore tipo:
```
error: unknown architecture: gemma3
```

**Soluzione**: llama.cpp potrebbe non supportare ancora Gemma 3.
- Usa `ui_gradio.py` (CPU, transformers, funzionante)
- Aspetta update llama.cpp
- O usa Gemma 2 / Llama 3

---

Testa appena la compilazione finisce! 🚀
