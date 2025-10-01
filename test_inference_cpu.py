"""
Test di inferenza del modello fine-tuned su CPU (più stabile)
Carica il modello base + adapter LoRA senza quantizzazione
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

# Leggi token HuggingFace
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN non trovato nelle variabili d'ambiente (.env)")


# Test prompts
TEST_PROMPTS = [
    "Tell me about Lewis Hamilton's performance in F1.",
    "What do you know about McLaren's lap times?",
    "Who typically finishes in position 1 at Monaco Grand Prix?",
]

# ==============================================================================
# Banner
# ==============================================================================
print("="*70)
print("  🧪 TEST INFERENZA CPU - Modello Fine-tuned F1")
print("="*70)
print()
print("⚙️ Configurazione:")
print(f"  Base Model: {BASE_MODEL_NAME}")
print(f"  Adapter: {ADAPTER_PATH}")
print(f"  Device: CPU (più stabile per inferenza)")
print()

# ==============================================================================
# Caricamento modello
# ==============================================================================

# 1. Tokenizer
print("📥 Caricamento tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)

# Fix pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"✓ Tokenizer caricato (pad_token_id={tokenizer.pad_token_id})")
print()

# 2. Modello base
print("📥 Caricamento modello base...")
print("   (Su CPU ci vorranno 3-5 minuti...)")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)

# Configura pad_token_id nel modello
model.config.pad_token_id = tokenizer.pad_token_id

print("✓ Modello base caricato")
print()

# 3. Carica adapter LoRA
print("📥 Caricamento adapter LoRA...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()  # Modalità inferenza
print("✓ Adapter caricato e merged")
print()

# ==============================================================================
# Test inferenza
# ==============================================================================
print("="*70)
print("  🧪 TESTING INFERENCE")
print("="*70)
print()

for i, prompt in enumerate(TEST_PROMPTS, 1):
    print(f"\n{'─'*70}")
    print(f"Test {i}/{len(TEST_PROMPTS)}")
    print(f"{'─'*70}")
    print(f"\n❓ Prompt: {prompt}\n")
    print(f"💬 Risposta:")
    
    try:
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate (su CPU)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,  # Ridotto per velocità su CPU
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Rimuovi il prompt dall'output
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        print(f"   {response}\n")
        
    except Exception as e:
        print(f"   ❌ Errore: {e}\n")

print()
print("="*70)
print("  ✅ TEST COMPLETATO")
print("="*70)
print()
print("📝 Note:")
print("  - Inferenza su CPU è più lenta ma più stabile")
print("  - Per velocità su GPU, serve risolvere il problema CUDA")
print("  - L'adapter è stato caricato correttamente")
print()
