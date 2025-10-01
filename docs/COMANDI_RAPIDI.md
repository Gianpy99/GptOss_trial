# 🚀 COMANDI RAPIDI - Fine-Tuning con Ollama_wrapper

## Test & Verifica

```powershell
# Test integrazione base
python test_integration.py

# Verifica che Ollama funzioni ancora
python test_wrapper.py
```

## Installazione Dipendenze

```powershell
# Dipendenze base (già installate)
pip install -r requirements.txt

# Dipendenze fine-tuning (NUOVO)
pip install -r requirements-finetuning.txt

# Oppure come optional dependency
pip install -e .[finetuning]
```

## Quick Start

```powershell
# Esempio veloce (5 minuti)
python examples/quick_start_finetuning.py

# Demo Formula 1 (prima/dopo fine-tuning)
python demo_f1_finetuning.py        # Demo completa interattiva
python test_f1_quick.py             # Test veloce F1

# Workflow completo interattivo
python examples/example_finetuning_integration.py

# Esegui step specifici
python examples/example_finetuning_integration.py step1  # Crea dati
python examples/example_finetuning_integration.py step2  # Fine-tune
python examples/example_finetuning_integration.py step3  # Testa
python examples/example_finetuning_integration.py step4  # Info integrazione
python examples/example_finetuning_integration.py full   # Pipeline completa
```

## Uso in Python

### Importa e Usa
```python
# Import base (sempre funziona)
from ollama_wrapper import OllamaWrapper, MemoryManager

# Import fine-tuning (se dipendenze installate)
from ollama_wrapper import FineTuningManager, create_finetuned_assistant

# Crea wrapper
wrapper = OllamaWrapper()

# Fine-tune
manager = FineTuningManager()
manager.load_model()
manager.setup_lora()
dataset = manager.load_training_data_from_memory()
tokenized = manager.tokenize_dataset(dataset)
manager.train(tokenized)
manager.save_adapter("my_adapter")

# Usa adapter
assistant = create_finetuned_assistant("./fine_tuned_models/my_adapter")
response = assistant.generate_text("User: Hello\n\nAssistant:")
```

## Documentazione

```powershell
# Leggi documentazione
notepad FINETUNING_SUMMARY.md      # Quick overview
notepad FINETUNING_GUIDE.md        # Guida completa
notepad INTEGRAZIONE_FATTA.md      # Cosa è stato fatto
notepad README.md                  # README aggiornato
```

## File Importanti

```
src/ollama_wrapper/
  ├── finetuning.py              # Modulo fine-tuning principale
  ├── wrapper.py                 # Wrapper Ollama originale
  └── __init__.py                # Exports

examples/
  ├── quick_start_finetuning.py                 # Quick start
  └── example_finetuning_integration.py         # Workflow completo

FINETUNING_GUIDE.md              # Guida completa
FINETUNING_SUMMARY.md            # Riepilogo veloce
INTEGRAZIONE_FATTA.md            # Cosa è stato aggiunto
requirements-finetuning.txt      # Dipendenze opzionali
test_integration.py              # Test integrazione
```

## Modelli Suggeriti

### Piccoli (Test/Sviluppo)
- `microsoft/phi-2` (2.7B) - Veloce, ottimo per test
- `microsoft/phi-1_5` (1.3B) - Ancora più piccolo

### Medi (Produzione)
- `meta-llama/Llama-2-7b-hf` (7B) - Equilibrato
- `mistralai/Mistral-7B-v0.1` (7B) - Ottimo rapporto qualità/velocità

### Grandi (Qualità Max)
- `meta-llama/Llama-2-13b-hf` (13B) - Più capacità
- `mistralai/Mixtral-8x7B-v0.1` - MoE, molto potente

## Configurazioni Rapide

### Per Test Veloci
```python
manager.setup_lora(r=8, lora_alpha=16)
manager.train(num_epochs=2, batch_size=2)
```

### Per Produzione
```python
manager.setup_lora(r=16, lora_alpha=32)
manager.train(num_epochs=3, batch_size=4, gradient_accumulation_steps=4)
```

### Per Qualità Massima
```python
manager.setup_lora(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"])
manager.train(num_epochs=5, batch_size=4, learning_rate=2e-4)
```

## Troubleshooting Veloce

```powershell
# Out of memory?
# → Riduci batch_size, abilita use_4bit=True

# Dipendenze non trovate?
pip install -r requirements-finetuning.txt

# Errori import?
python test_integration.py

# Training troppo lento?
# → Usa GPU, abilita 4-bit, riduci batch_size

# Nessun dato di training?
# → Crea conversazioni prima con OllamaWrapper
```

## Link Utili

- Hugging Face Models: https://huggingface.co/models
- PEFT Documentation: https://huggingface.co/docs/peft
- LoRA Paper: https://arxiv.org/abs/2106.09685
- QLoRA Paper: https://arxiv.org/abs/2305.14314

## Prossimi Passi

1. ✅ Installa dipendenze
2. ✅ Testa: `python test_integration.py`
3. ✅ Quick start: `python examples/quick_start_finetuning.py`
4. ✅ Leggi: `FINETUNING_GUIDE.md`
5. ✅ Crea il tuo primo adapter!

---
**Happy Fine-Tuning! 🎉**
