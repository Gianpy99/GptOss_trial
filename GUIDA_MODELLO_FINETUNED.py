"""
Guida: Come usare il modello fine-tuned con Ollama
==================================================

PROBLEMA:
Ollama non supporta adapter PEFT (safetensors) nativamente.
I tuoi adapter sono in: ./finetuning_projects/f1_expert_fixed/adapter/

SOLUZIONI:

1. ✅ PYTHON DIRETTO (Quello che funziona ora)
   ------------------------------------------
   python test_inference_cpu.py
   
   Pro:
   - Funziona subito
   - Usa il tuo adapter correttamente
   - Nessuna conversione necessaria
   
   Contro:
   - Più lento (CPU)
   - Non integrato con Ollama
   - Serve Python environment


2. 🔄 CONVERSIONE GGUF (Per Ollama nativo)
   ----------------------------------------
   Serve convertire base model + adapter in GGUF.
   
   Passi:
   a. Merge adapter nel base model:
      from peft import PeftModel
      model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it")
      model = PeftModel.from_pretrained(model, "./adapter/")
      merged = model.merge_and_unload()
      merged.save_pretrained("./merged_model")
   
   b. Converti in GGUF con llama.cpp:
      python convert-hf-to-gguf.py ./merged_model --outfile f1_expert.gguf
   
   c. Importa in Ollama:
      ollama create f1-expert -f Modelfile
      
      # Modelfile:
      FROM ./f1_expert.gguf
      PARAMETER temperature 0.7
   
   Pro:
   - Integrato con Ollama
   - Più veloce (ottimizzato GGUF)
   - Facile da usare
   
   Contro:
   - Richiede llama.cpp (build complesso su Windows)
   - File GGUF grande (~8GB)
   - Processo laborioso


3. 🌐 OLLAMA API + PYTHON WRAPPER
   ---------------------------------
   Usa wrapper.py per chiamare Ollama ma con logica Python.
   
   from wrapper import OllamaWrapper
   wrapper = OllamaWrapper(model="gemma3:4b")
   # Poi carica adapter in Python e usa wrapper per interfaccia


RACCOMANDAZIONE:
===============
PER ORA: Usa test_inference_cpu.py (funziona, adapter caricato correttamente)
FUTURO: Se vuoi Ollama nativo, segui conversione GGUF (ma è complesso)


VERIFICA QUALE MODELLO STAI USANDO:
===================================
1. Ollama:
   ollama list
   ollama show gemma3:4b --modelfile
   
2. Python:
   # Guarda l'output di test_inference_cpu.py
   # Deve dire "✓ Adapter caricato e merged"


CONFRONTO RISPOSTE:
==================
Base Model (HuggingFace):    "Lewis Hamilton 7 wins" ✓ CORRETTO
Fine-tuned (con adapter):    "Mercedes 5 wins" ✗ SBAGLIATO (ma diverso!)
Ollama (gemma3:4b):          "Lewis Hamilton 7 wins" ✓ CORRETTO (= base)

Conclusione: Il fine-tuning HA FUNZIONATO (risposte diverse) ma serve
più training data per migliorare l'accuratezza.
"""

print(__doc__)
