# 🏎️ Demo F1 Fine-Tuning - Guida Rapida

## ✅ File Creati

1. **`demo_f1_finetuning.py`** - Demo completa interattiva
   - Scarica dataset F1
   - Crea training data
   - Testa PRIMA del fine-tuning
   - Esegue fine-tuning
   - Testa DOPO il fine-tuning
   - Confronta i risultati

2. **`test_f1_quick.py`** - Test veloce
   - Versione semplificata
   - Training più veloce
   - Ideale per test rapidi

3. **`F1_DEMO_README.md`** - Documentazione completa
   - Spiegazione dettagliata
   - Configurazione
   - Troubleshooting
   - Personalizzazione

4. **`test_f1_dataset.py`** - Test connessione dataset

## 🚀 Come Eseguire

### Opzione 1: Demo Completa (Raccomandato)

```powershell
python demo_f1_finetuning.py
```

**Cosa succede:**
1. Scarica 100 righe da Hugging Face Formula_1_Dataset
2. Crea ~50 esempi di training (domande su piloti/team/tempi)
3. Carica modello base (phi-2)
4. Fa 4 domande al modello base → Risposte generiche/errate
5. Fine-tune il modello (3 epoche, ~10 minuti)
6. Fa le STESSE 4 domande al modello fine-tuned → Risposte accurate!
7. Mostra confronto prima/dopo

**Domande di Test:**
- "What team does VER drive for in Formula 1?"
- "Which drivers race for Red Bull Racing?"
- "What is VER's average lap time?"
- "Who is the fastest driver based on lap times?"

### Opzione 2: Test Veloce

```powershell
python test_f1_quick.py
```

**Più veloce (~5 min):**
- Solo 20 esempi
- 2 epoche
- Configurazione LoRA più leggera

## 📊 Risultati Attesi

### PRIMA del Fine-Tuning
```
Q: What team does VER drive for in Formula 1?
A: VER could refer to various things. Without more context,
   I cannot provide a specific answer about Formula 1 teams...
```
*(Risposta generica, non sa chi è VER)*

### DOPO il Fine-Tuning
```
Q: What team does VER drive for in Formula 1?
A: VER drives for Red Bull Racing in Formula 1.
```
*(Risposta precisa e accurata!)*

## 🎯 Dimostrazione Chiave

La demo dimostra che:

1. **Specializzazione**: Il modello impara conoscenze specifiche (F1)
2. **Efficienza**: Adapter di soli ~20 MB vs 3 GB modello completo
3. **Velocità**: Training in ~10 minuti con dati reali
4. **Praticità**: Usa dati pubblici da Hugging Face

## 🔧 Requisiti

```powershell
# Installa dipendenze (se non fatto)
pip install -r requirements-finetuning.txt

# Verifica dataset accessibile
python test_f1_dataset.py
```

**Hardware:**
- GPU raccomandato (RTX 3090: ~5 min, GTX 1080: ~15 min)
- ~8 GB RAM GPU con QLoRA 4-bit
- CPU possibile ma molto più lento

## 📁 Output Generato

Dopo l'esecuzione troverai:

```
f1_training_data.json              # Dati di training generati
fine_tuned_models/
  └── f1_expert_adapter/           # Adapter fine-tuned (~20 MB)
      ├── adapter_config.json
      ├── adapter_model.bin
      └── ...
```

## 🔄 Riusare l'Adapter

```python
from ollama_wrapper import create_finetuned_assistant

# Carica l'adapter F1
f1_expert = create_finetuned_assistant(
    "./fine_tuned_models/f1_expert_adapter",
    temperature=0.7
)

# Fai domande su F1
response = f1_expert.generate_text(
    "User: Tell me about Max Verstappen's team\n\nAssistant:"
)
print(response)
```

## 💡 Personalizzazione

### Più Dati
```python
# In demo_f1_finetuning.py, riga ~50:
rows = download_f1_dataset(limit=200)  # Aumenta a 200
```

### Diverse Domande
```python
# Modifica test_questions (riga ~185):
test_questions = [
    "What is the average track temperature?",
    "How does rainfall affect performance?",
    "Which team has the fastest average lap time?",
]
```

### Training Più Lungo
```python
# In finetune_on_f1_data(), riga ~245:
manager.train(
    num_epochs=5,        # Aumenta epoche
    batch_size=4,        # Aumenta batch se hai memoria
)
```

## 🐛 Troubleshooting

### Dataset Non Scaricato
```
❌ Error downloading dataset
```
→ Controlla internet, riprova dopo qualche minuto

### Out of Memory
```
❌ CUDA out of memory
```
→ Usa `test_f1_quick.py` (configurazione più leggera)
→ Oppure riduci batch_size a 1

### Training Lento
→ Normale su CPU/GPU lente
→ Usa `test_f1_quick.py` per versione veloce
→ Riduci num_epochs a 2

## 📚 Documentazione Completa

Leggi `F1_DEMO_README.md` per:
- Spiegazione dettagliata del codice
- Struttura dataset
- Configurazioni avanzate
- Use case reali
- Best practices

## 🎯 Prossimi Passi

1. ✅ Esegui la demo: `python demo_f1_finetuning.py`
2. ✅ Osserva il miglioramento prima/dopo
3. ✅ Leggi `F1_DEMO_README.md` per dettagli
4. ✅ Personalizza con tue domande
5. ✅ Applica lo stesso approccio al tuo dominio!

## 🌟 Applicabilità

Questo approccio funziona per QUALSIASI dominio:
- 📊 Dati finanziari
- 🏥 Dati medici
- 📚 Knowledge base aziendali
- 🛒 Cataloghi prodotti
- 📝 Documentazione tecnica
- 💬 FAQ e supporto
- ... qualsiasi dataset strutturato!

## 🏁 Esegui Ora!

```powershell
# Demo completa (10-15 min)
python demo_f1_finetuning.py

# O test veloce (5 min)
python test_f1_quick.py
```

**Preparati a vedere la magia del fine-tuning! 🏎️💨**

---

## 📞 Hai Problemi?

1. Controlla `F1_DEMO_README.md` → Sezione Troubleshooting
2. Verifica requisiti: `python test_integration.py`
3. Testa dataset: `python test_f1_dataset.py`
4. Leggi errori nel terminale

## ✨ Risultato Finale

Avrai dimostrato che:
- ✅ Il modello base NON conosce i dettagli F1
- ✅ Dopo fine-tuning, il modello è un esperto F1
- ✅ Tutto con solo ~20 MB di adapter
- ✅ Training in pochi minuti
- ✅ Approach riutilizzabile per qualsiasi dominio

**Buon fine-tuning! 🚀**
