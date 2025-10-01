"""
Test Inference con Adapter LoRA (diretto, senza Ollama)
Carica il modello base + adapter LoRA e testa l'inferenza
"""

import os
import sys
import io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Forza UTF-8 su Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print(f"\n{'='*70}")
print(f"  🧪 TEST DIRECT INFERENCE - Fine-tuned Model")
print(f"{'='*70}\n")

# Configurazione
BASE_MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_PATH = "./finetuning_projects/f1_expert_fixed/adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load token
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"⚙️ Configurazione:")
print(f"  Base Model: {BASE_MODEL_NAME}")
print(f"  Adapter Path: {ADAPTER_PATH}")
print(f"  Device: {DEVICE}")
print()

# Verifica adapter
if not os.path.exists(ADAPTER_PATH):
    print(f"❌ Errore: Adapter non trovato in {ADAPTER_PATH}")
    sys.exit(1)

print(f"✓ Adapter trovato")
print()

# Test prompts
TEST_PROMPTS = [
    "Tell me about Lewis Hamilton's performance in F1.",
    "What do you know about McLaren's lap times?",
    "Who typically finishes in position 1 at Monaco Grand Prix?",
]

print(f"📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
print(f"✓ Tokenizer loaded")
print()

print(f"📥 Loading base model...")
print(f"   (Questo può richiedere 2-3 minuti...)")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True,  # Usa quantizzazione per ridurre memoria
    token=HF_TOKEN
)
print(f"✓ Base model loaded")
print()

print(f"📥 Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
print(f"✓ Adapter loaded and merged")
print()

# Test inference
print(f"{'='*70}")
print(f"  🧪 TESTING INFERENCE")
print(f"{'='*70}\n")

for i, prompt in enumerate(TEST_PROMPTS, 1):
    print(f"\n{'─'*70}")
    print(f"Test {i}/{len(TEST_PROMPTS)}")
    print(f"{'─'*70}")
    print(f"\n❓ Prompt: {prompt}\n")
    print(f"💬 Risposta:")
    
    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Rimuovi il prompt dall'output
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        print(f"   {response}\n")
        
    except Exception as e:
        print(f"   ❌ Errore: {e}\n")

print(f"\n{'='*70}")
print(f"  ✅ TEST COMPLETATO")
print(f"{'='*70}\n")

print(f"📝 Note:")
print(f"  - Questo test usa il modello + adapter direttamente (senza Ollama)")
print(f"  - Per usare con Ollama, serve convertire in formato GGUF")
print(f"  - Alternativamente, usa questo script per inferenza Python diretta")
print()

print(f"💡 Prossimi passi:")
print(f"  1. Confronta le risposte con il modello base (senza adapter)")
print(f"  2. Verifica se menziona dati specifici dal training (lap times, posizioni)")
print(f"  3. Per deployment su Ollama, considera conversione a GGUF")
print()
