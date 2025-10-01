# 🎨 Guida UI Web Semplice

## 🚀 Quick Start (3 Minuti)

### 1. Installa Gradio
```bash
.\.venv_training\Scripts\Activate.ps1
pip install gradio
```

### 2. Avvia UI
```bash
python ui_simple.py
```

### 3. Usa l'interfaccia
- Il browser si apre automaticamente su http://localhost:7860
- Scrivi la tua domanda F1
- Clicca "Invia" o premi Enter
- Regola temperatura e lunghezza risposta con gli slider

---

## ✨ Caratteristiche

✅ **Semplice**: Solo Python + browser, nessun server complesso
✅ **Locale**: Gira solo sul tuo PC (non online)
✅ **Veloce**: 2-3 minuti caricamento, poi istantaneo
✅ **Controllabile**: Slider per temperatura e lunghezza
✅ **Chat History**: Mantiene la conversazione
✅ **Esempi**: Click su esempi predefiniti

---

## ⚙️ Personalizzazione

### Usa il Modello Base (senza fine-tuning)
Modifica `ui_simple.py` riga 18:
```python
USE_BASE_MODEL = True  # Cambia da False a True
```

### Cambia Porta
Modifica riga 185:
```python
server_port=8080,  # Invece di 7860
```

### Aggiungi Esempi
Modifica riga 96:
```python
examples = [
    ["La tua domanda custom..."],
    ["Un'altra domanda..."],
]
```

---

## 🎛️ Parametri

### Temperature (0.1 - 1.0)
- **0.1-0.3**: Risposte precise e deterministiche
- **0.5-0.7**: Bilanciato (default)
- **0.8-1.0**: Creativo e variabile

### Max Tokens (50 - 500)
- **50-100**: Risposte brevi
- **150-250**: Risposte medie (default 200)
- **300-500**: Risposte lunghe e dettagliate

---

## 🔍 Screenshot (cosa vedrai)

```
┌─────────────────────────────────────────────────┐
│  🏎️ F1 Expert Assistant                        │
│  Modello Fine-tuned su Dati Formula 1          │
├─────────────────────────────────────────────────┤
│                                                 │
│  👤 You: Tell me about Lewis Hamilton          │
│                                                 │
│  🤖 Assistant: Lewis Hamilton is one of the    │
│     most successful Formula 1 drivers...       │
│                                                 │
├─────────────────────────────────────────────────┤
│  [Il tuo messaggio...]              [Invia]    │
└─────────────────────────────────────────────────┘
     Temperature: ●────────── 0.7
     Max Tokens:  ●────────── 200
     
     📝 Esempi:
     • Tell me about Lewis Hamilton
     • McLaren lap times
     • Monaco GP position 1
```

---

## 🐛 Troubleshooting

### Porta già in uso
```bash
# Cambia porta in ui_simple.py
server_port=8080  # O qualsiasi numero 7000-9000
```

### Modello non caricato
Verifica che esista:
```bash
dir fine_tuned_models\f1_expert_merged
```

Se non c'è:
```bash
python merge_adapter.py
```

### Gradio non installato
```bash
pip install gradio --upgrade
```

### Out of Memory (CPU)
Il modello richiede ~8 GB RAM. Se hai problemi:
1. Chiudi altri programmi
2. O usa quantizzazione (modifica ui_simple.py):
```python
torch_dtype=torch.float16  # Invece di float32
```

---

## 🔄 Confronto con Altre Opzioni

| Caratteristica | ui_simple.py | LM Studio | Text-Gen WebUI | Python CLI |
|----------------|--------------|-----------|----------------|------------|
| Setup | 🟢 2 min | 🟢 5 min | 🟡 15 min | 🟢 0 min |
| UI Grafica | ✅ Web | ✅ Desktop | ✅ Web | ❌ Terminal |
| Complessità | 🟢 Bassa | 🟢 Bassa | 🔴 Alta | 🟢 Bassa |
| Supporto Gemma3 | ✅ Sì | ❌ No | ✅ Sì | ✅ Sì |
| Condivisione | ❌ Solo locale | ❌ Solo locale | ✅ Possibile | ❌ Solo locale |
| Personalizzabile | ✅ Molto | ❌ Poco | ✅ Molto | ✅ Molto |

---

## 💡 Pro Tips

### 1. Condividi Temporaneamente
Modifica in `demo.launch()`:
```python
share=True  # Crea link pubblico temporaneo (7 giorni)
```

### 2. Salva Conversazioni
Aggiungi bottone "Salva":
```python
save_btn = gr.Button("💾 Salva Chat")
save_btn.click(save_conversation, inputs=chatbot, outputs=file_output)
```

### 3. Modalità Streaming
Per vedere la risposta mentre viene generata (avanzato):
```python
# Sostituisci model.generate con streaming generator
```

---

## 🎯 Next Steps

Dopo che l'UI funziona:

1. **Migliora il training**:
   - Usa dataset con dati più specifici
   - Più epoche (5-10 invece di 3)
   
2. **Aggiungi funzionalità**:
   - Salvataggio chat
   - Upload file
   - Text-to-speech per le risposte
   
3. **Deploy (opzionale)**:
   - Hugging Face Spaces (gratis)
   - Render.com (gratis)
   - Docker container

---

Pronti? Esegui:
```bash
.\.venv_training\Scripts\Activate.ps1
pip install gradio
python ui_simple.py
```
