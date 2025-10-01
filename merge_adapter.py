"""
MERGE Adapter nel Base Model
Crea un modello COMPLETO da caricare in Ollama/LM Studio

Questo script:
1. Carica google/gemma-3-4b-it (base model)
2. Carica il tuo adapter LoRA
3. Fa il MERGE (unisce i weights)
4. Salva il modello COMPLETO in formato HuggingFace

Dopo puoi convertire in GGUF per Ollama/LM Studio
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

# ==============================================================================
# Configurazione
# ==============================================================================
BASE_MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_PATH = "./finetuning_projects/f1_expert_fixed/adapter"
OUTPUT_PATH = "./fine_tuned_models/f1_expert_merged"  # Dove salvare il modello merged

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

print("="*70)
print("  🔄 MERGE ADAPTER NEL BASE MODEL")
print("="*70)
print()
print("⚙️ Configurazione:")
print(f"  Base Model: {BASE_MODEL_NAME}")
print(f"  Adapter: {ADAPTER_PATH}")
print(f"  Output: {OUTPUT_PATH}")
print()

# ==============================================================================
# Step 1: Carica Base Model
# ==============================================================================
print("📥 Step 1/4: Caricamento base model...")
print("   (Questo richiede 2-3 minuti...)")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,  # Usa float16 per risparmiare spazio
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)
print("✓ Base model caricato")
print()

# ==============================================================================
# Step 2: Carica Tokenizer
# ==============================================================================
print("📥 Step 2/4: Caricamento tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer caricato")
print()

# ==============================================================================
# Step 3: Carica Adapter e MERGE
# ==============================================================================
print("📥 Step 3/4: Caricamento adapter e MERGE...")
print("   (Questo crea un modello COMPLETO senza dipendenze da adapter)")

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()  # ← QUESTO FA IL MERGE!

print("✓ Adapter merged nel modello base")
print()

# ==============================================================================
# Step 4: Salva Modello Merged
# ==============================================================================
print("💾 Step 4/4: Salvataggio modello merged...")
print(f"   Percorso: {OUTPUT_PATH}")

# Crea la directory se non esiste
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Salva il modello
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✓ Modello salvato!")
print()

# ==============================================================================
# Informazioni finali
# ==============================================================================
print("="*70)
print("  ✅ MERGE COMPLETATO")
print("="*70)
print()
print(f"📂 Modello salvato in: {OUTPUT_PATH}")
print()
print("📊 Cosa hai ora:")
print("  - Modello COMPLETO (base + fine-tuning)")
print("  - Formato: HuggingFace safetensors")
print("  - Dimensione: ~8 GB (float16)")
print()
print("🔄 Prossimi passi per usarlo:")
print()
print("  1️⃣  OLLAMA (richiede conversione GGUF):")
print("     - Serve llama.cpp per convertire in GGUF")
print("     - Windows: complesso (serve build llama.cpp)")
print("     - Alternativa: usa llama-cpp-python")
print()
print("  2️⃣  LM STUDIO:")
print("     - File > Import > Seleziona la cartella:")
print(f"       {os.path.abspath(OUTPUT_PATH)}")
print("     - LM Studio può caricare safetensors direttamente")
print()
print("  3️⃣  PYTHON (quello che usi ora):")
print("     # Usa il merged model invece di base + adapter")
print("     model = AutoModelForCausalLM.from_pretrained(")
print(f'         "{OUTPUT_PATH}")')
print()
print("💡 NOTA: Per Ollama serve GGUF. Vuoi che crei lo script di conversione?")
print()
