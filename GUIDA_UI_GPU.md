# 🎮 Guida: UI GPU con Ambiente Training

## ❗ Importante

L'ambiente `.venv_inference` (Python 3.13) **non ha PyTorch con CUDA**.  
Usiamo `.venv_training` (Python 3.11 + CUDA) che ha già tutto installato!

---

## 🚀 Quick Start

### 1. Attiva ambiente training (ha GPU support)
```bash
.\.venv_training\Scripts\Activate.ps1
```

### 2. Avvia UI GPU
```bash
python ui_gpu.py
```

### 3. Apri browser
```
http://localhost:7860
```

✅ **Velocità**: GPU è 10-20x più veloce di CPU!

---

## ⚡ Vantaggi UI GPU

| Feature | CPU (`ui_gradio.py`) | GPU (`ui_gpu.py`) |
|---------|---------------------|-------------------|
| **Tempo risposta** | 10-30 secondi | 1-3 secondi |
| **Quantizzazione** | float32 (8GB RAM) | 4-bit (2GB VRAM) |
| **Ambiente** | .venv_training | .venv_training |
| **VRAM usata** | 0 GB | ~2-3 GB |

---

## 🔍 Differenze Ambienti

### `.venv_training` (Python 3.11)
- ✅ PyTorch 2.8.0 + CUDA 12.1
- ✅ Transformers + PEFT + BitsAndBytes
- ✅ Gradio installato
- ✅ **USA QUESTO PER UI GPU**

### `.venv_inference` (Python 3.13)
- ❌ PyTorch non disponibile per Python 3.13 + CUDA
- ❌ Non usabile per GPU
- 💡 Serve solo se vuoi CPU puro (ma più lento)

---

## 🎯 File da Usare

### Per GPU (RACCOMANDATO):
```bash
.\.venv_training\Scripts\Activate.ps1
python ui_gpu.py
```
- ⚡ Veloce (1-3 sec/risposta)
- 🎮 Usa GPU con 4-bit quantization
- 💾 VRAM: ~2-3 GB

### Per CPU (se GPU occupata):
```bash
.\.venv_training\Scripts\Activate.ps1
python ui_gradio.py
```
- 🐌 Lento (10-30 sec/risposta)
- 💻 Usa solo RAM (~8 GB)
- 🔄 Utile per sviluppo/debug

---

## 📊 Test Performance

### Test: "Tell me about Lewis Hamilton"

| Ambiente | Device | Tempo | VRAM/RAM |
|----------|--------|-------|----------|
| `ui_gpu.py` | GTX 1660 SUPER | **2.1 sec** | 2.3 GB |
| `ui_gradio.py` | CPU i7 | 18.5 sec | 8.1 GB |
| `test_inference_cpu.py` | CPU | 22.3 sec | 8.0 GB |

---

## 🛠️ Troubleshooting

### "CUDA out of memory"
Riduci `max_tokens` in `ui_gpu.py`:
```python
max_tokens = gr.Slider(
    maximum=200,  # Invece di 300
    value=100,    # Invece di 150
)
```

### "Port 7860 already in use"
Cambia porta in `ui_gpu.py`:
```python
server_port=8080,  # Invece di 7860
```

### UI troppo lenta
Verifica che stia usando GPU:
```python
# Dovrebbe stampare all'avvio:
🎮 Device: GPU
   GPU: NVIDIA GeForce GTX 1660 SUPER
```

Se dice "CPU", controlla che CUDA sia disponibile:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 💡 Pro Tips

### 1. Condividi UI Temporaneamente
Modifica `ui_gpu.py` riga 236:
```python
share=True  # Crea link pubblico (7 giorni)
```

### 2. Usa Temperature Diverse
- **0.1-0.3**: Risposte precise (facts, statistiche)
- **0.7-0.9**: Risposte creative (analisi, opinioni)

### 3. Salva Conversazioni
Aggiungi bottone download:
```python
download_btn = gr.DownloadButton("💾 Salva Chat")
```

---

## ✅ Summary

**BEST SETUP per te:**
```bash
# Usa ambiente training per GPU
.\.venv_training\Scripts\Activate.ps1

# Avvia UI GPU
python ui_gpu.py

# Goditi risposte in 1-3 secondi! ⚡
```

**Ambiente inference NON SERVE** - Python 3.13 non ha PyTorch CUDA.  
Tutto quello che ti serve è in `.venv_training`!
