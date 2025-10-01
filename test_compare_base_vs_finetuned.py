"""
Confronto Base Model vs Fine-tuned Model
Testa gli stessi prompt su entrambi per vedere le differenze
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

# Config
BASE_MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_PATH = "./finetuning_projects/f1_expert_fixed/adapter"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

TEST_PROMPTS = [
    "Tell me about Lewis Hamilton's performance in F1.",
    "What do you know about McLaren's lap times?",
    "Who typically finishes in position 1 at Monaco Grand Prix?",
]

print("="*70)
print("  🔬 COMPARAZIONE: Base Model vs Fine-tuned Model")
print("="*70)
print()

# Carica tokenizer
print("📥 Caricamento tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print("✓ Tokenizer caricato")
print()

# Carica modello base
print("📥 Caricamento modello BASE (senza fine-tuning)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.eval()
print("✓ Base model caricato")
print()

# Carica modello fine-tuned
print("📥 Caricamento modello FINE-TUNED (con adapter)...")
finetuned_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)
finetuned_model.config.pad_token_id = tokenizer.pad_token_id
finetuned_model = PeftModel.from_pretrained(finetuned_model, ADAPTER_PATH)
finetuned_model.eval()
print("✓ Fine-tuned model caricato")
print()

# Test comparativo
print("="*70)
print("  🧪 TESTING COMPARATIVO")
print("="*70)

for i, prompt in enumerate(TEST_PROMPTS, 1):
    print(f"\n{'='*70}")
    print(f"Test {i}/{len(TEST_PROMPTS)}")
    print(f"{'='*70}")
    print(f"\n❓ Prompt: {prompt}\n")
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Base model
    print("📝 BASE MODEL (senza fine-tuning):")
    with torch.no_grad():
        outputs_base = base_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_base = tokenizer.decode(outputs_base[0], skip_special_tokens=True)
    if response_base.startswith(prompt):
        response_base = response_base[len(prompt):].strip()
    print(f"   {response_base}\n")
    
    # Fine-tuned model
    print("🎯 FINE-TUNED MODEL (con adapter):")
    with torch.no_grad():
        outputs_ft = finetuned_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_ft = tokenizer.decode(outputs_ft[0], skip_special_tokens=True)
    if response_ft.startswith(prompt):
        response_ft = response_ft[len(prompt):].strip()
    print(f"   {response_ft}\n")
    
    # Evidenzia differenze
    if response_base.lower() != response_ft.lower():
        print("   ✓ Le risposte sono DIVERSE (fine-tuning ha effetto)")
    else:
        print("   ⚠️  Le risposte sono IDENTICHE (fine-tuning ha poco effetto)")

print()
print("="*70)
print("  ✅ COMPARAZIONE COMPLETATA")
print("="*70)
print()
print("📊 Conclusioni:")
print("  - Se le risposte sono diverse: fine-tuning ha modificato il comportamento")
print("  - Se sono identiche: serve più training data o più epoche")
print()
