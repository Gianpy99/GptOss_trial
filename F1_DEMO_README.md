# 🏎️ Formula 1 Fine-Tuning Demo

## Descrizione

Questa demo dimostra in modo pratico come il fine-tuning migliora le prestazioni di un modello su un dominio specifico (Formula 1).

## 🎯 Obiettivo

Mostrare che:
1. **PRIMA del fine-tuning**: Il modello dà risposte generiche o errate
2. **DOPO il fine-tuning**: Il modello fornisce risposte accurate basate sui dati F1

## 📁 File

- **`demo_f1_finetuning.py`** - Demo completa interattiva (~400 righe)
  - Scarica dataset F1 da Hugging Face
  - Crea dati di training strutturati
  - Confronta risposte prima/dopo
  - Include pause per review

- **`test_f1_quick.py`** - Test veloce (~150 righe)
  - Versione semplificata
  - Più veloce (2 epoche, meno dati)
  - Ideale per test rapidi

## 🚀 Come Usare

### Demo Completa (Raccomandato)

```powershell
# Esegui la demo completa
python demo_f1_finetuning.py
```

**Cosa fa:**
1. Scarica ~100 righe di dati F1 da Hugging Face
2. Crea ~50+ esempi di training (domande/risposte)
3. Testa modello base su 4 domande F1
4. Fine-tune il modello (3 epoche)
5. Ritesta le stesse 4 domande
6. Mostra confronto prima/dopo

**Output:**
- `f1_training_data.json` - Dati di training
- `./fine_tuned_models/f1_expert_adapter/` - Adapter fine-tuned

**Tempo:** ~10-15 minuti (dipende da GPU)

### Test Veloce

```powershell
# Test rapido
python test_f1_quick.py
```

**Differenze:**
- Solo primi 20 esempi
- 2 epoche invece di 3
- Batch size 1
- LoRA rank 8 (vs 16)

**Tempo:** ~5-8 minuti

## 📊 Dataset

**Source:** [Vadera007/Formula_1_Dataset](https://huggingface.co/datasets/Vadera007/Formula_1_Dataset)

**Campi:**
- `Driver` - Codice pilota (es. "VER", "PER")
- `Team` - Scuderia (es. "Red Bull Racing")
- `AvgLapTime` - Tempo medio giro (secondi)
- `LapsCompleted` - Giri completati
- `AirTemp` - Temperatura aria (°C)
- `TrackTemp` - Temperatura pista (°C)
- `Rainfall` - Pioggia (bool)
- `QualiPosition` - Posizione qualifica
- `RaceFinishPosition` - Posizione finale gara

## 🎓 Esempi di Training Generati

Il codice trasforma i dati raw in coppie instruction-output:

```json
{
  "instruction": "What team does VER drive for in Formula 1?",
  "output": "VER drives for Red Bull Racing in Formula 1."
}
```

```json
{
  "instruction": "What is VER's average lap time?",
  "output": "VER's average lap time is approximately 90.35 seconds."
}
```

```json
{
  "instruction": "Which drivers race for Red Bull Racing?",
  "output": "The drivers racing for Red Bull Racing are: PER, VER."
}
```

## 📈 Risultati Attesi

### PRIMA del Fine-Tuning

**Q:** What team does VER drive for?  
**A:** *(risposta generica/errata)* "VER could refer to various abbreviations. In Formula 1 context, without more specific information, I cannot provide a definitive answer..."

### DOPO il Fine-Tuning

**Q:** What team does VER drive for?  
**A:** *(risposta accurata)* "VER drives for Red Bull Racing in Formula 1."

## 🔧 Configurazione Fine-Tuning

### Demo Completa
```python
LoRA:
  - r=16
  - lora_alpha=32
  - lora_dropout=0.05

Training:
  - num_epochs=3
  - batch_size=2
  - learning_rate=2e-4
  - gradient_accumulation_steps=4
```

### Test Veloce
```python
LoRA:
  - r=8
  - lora_alpha=16

Training:
  - num_epochs=2
  - batch_size=1
  - learning_rate=2e-4
```

## 💡 Personalizzazione

### Usare Più Dati
```python
# In demo_f1_finetuning.py, modifica:
rows = download_f1_dataset(limit=100)  # Aumenta a 200, 500, etc.
```

### Altre Domande
```python
# Aggiungi domande in test_questions:
test_questions = [
    "What team does VER drive for?",
    "Who is the fastest driver?",
    "What are typical track temperatures?",  # ← NUOVA
    "How does rainfall affect lap times?",   # ← NUOVA
]
```

### Configurazione Training
```python
# Per training più aggressivo:
manager.setup_lora(r=32, lora_alpha=64)
manager.train(num_epochs=5, batch_size=4)

# Per training più veloce:
manager.setup_lora(r=8, lora_alpha=16)
manager.train(num_epochs=1, batch_size=1)
```

## 🐛 Troubleshooting

### Dataset Non Scaricato
```
❌ Error downloading dataset
```
**Soluzione:** Controlla connessione internet, riprova dopo qualche minuto

### Out of Memory
```
❌ CUDA out of memory
```
**Soluzioni:**
```python
# Opzione 1: Batch size più piccolo
manager.train(batch_size=1, gradient_accumulation_steps=8)

# Opzione 2: Meno dati
rows = rows[:30]  # Usa solo 30 esempi
```

### Training Lento
**Normale:** 5-15 minuti dipende da:
- GPU (RTX 3090: ~5 min, GTX 1080: ~15 min)
- Numero esempi
- Epoche

**Accelerare:**
- Riduci epoche a 2
- Riduci esempi di training
- Usa `test_f1_quick.py`

## 📝 Note

### Adapter Size
L'adapter finale è solo **~20-30 MB** invece di 3+ GB del modello completo!

### Riusabilità
```python
# Riusa l'adapter in altri script:
from ollama_wrapper import create_finetuned_assistant

f1_expert = create_finetuned_assistant(
    "./fine_tuned_models/f1_expert_adapter"
)

response = f1_expert.generate_text(
    "User: Tell me about Red Bull Racing\n\nAssistant:"
)
```

### Continual Learning
Puoi ri-fine-tune con nuovi dati:
1. Scarica dataset aggiornato
2. Crea nuovi training examples
3. Fine-tune di nuovo (partendo da adapter esistente o da zero)

## 🎯 Use Case Reali

Questo approccio funziona per:
- ✅ Dati sportivi (calcio, basket, etc.)
- ✅ Dati finanziari (stock, crypto)
- ✅ Dati aziendali interni
- ✅ Knowledge base specifiche
- ✅ FAQ e supporto clienti
- ✅ Documentazione tecnica

## 🔗 Link Utili

- Dataset: https://huggingface.co/datasets/Vadera007/Formula_1_Dataset
- PEFT Docs: https://huggingface.co/docs/peft
- Guida Fine-Tuning: [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md)

## ✅ Checklist Prima di Eseguire

- [ ] Dipendenze installate: `pip install -r requirements-finetuning.txt`
- [ ] GPU disponibile (raccomandato, ma non obbligatorio)
- [ ] ~8 GB RAM GPU libera (con QLoRA 4-bit)
- [ ] Connessione internet (per scaricare dataset)
- [ ] 10-15 minuti di tempo disponibili

## 🎉 Esegui la Demo!

```powershell
# Demo completa
python demo_f1_finetuning.py

# Oppure test veloce
python test_f1_quick.py
```

**Buon fine-tuning! 🏎️💨**
