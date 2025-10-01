"""
F1 Fine-tuning Comparison Test
Confronta le risposte del modello base vs fine-tuned
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

print(f"\n{'='*60}")
print(f"  F1 FINE-TUNING COMPARISON TEST")
print(f"{'='*60}\n")

# Load token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Parametri
MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_PATH = "./finetuning_projects/f1_test_gpu/adapter"

# Test questions
TEST_QUESTIONS = [
    "Who is Lewis Hamilton?",
    "Tell me about Max Verstappen's performance.",
    "What is Mercedes F1 team?",
    "Who drives for Red Bull Racing?",
    "Tell me about Ferrari F1 team."
]

print(f"Verifica GPU...")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}\n")

# Quantization config
print(f"Setup 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print(f"✓ Quantization ready\n")

# Load tokenizer
print(f"📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer loaded\n")

# Load base model
print(f"📥 Loading BASE model (questo richiederà 2-3 minuti)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)
print(f"✓ Base model loaded\n")

# Load fine-tuned model
print(f"📥 Loading FINE-TUNED model...")
finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
print(f"✓ Fine-tuned model loaded\n")

def generate_response(model, prompt, max_new_tokens=150):
    """Generate response from model"""
    # Formato semplice senza token speciali problematici
    formatted_prompt = f"Question: {prompt}\nAnswer:"
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        add_special_tokens=True
    ).to(model.device)
    
    # Assicura che pad_token_id sia valido
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decodifica solo i nuovi token generati (esclude prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()

print(f"{'='*60}")
print(f"  STARTING COMPARISON")
print(f"{'='*60}\n")

for i, question in enumerate(TEST_QUESTIONS, 1):
    print(f"\n{'─'*60}")
    print(f"🔹 TEST {i}/{len(TEST_QUESTIONS)}")
    print(f"{'─'*60}")
    print(f"\n❓ Question: {question}\n")
    
    # Base model response
    print(f"🤖 BASE MODEL:")
    base_response = generate_response(base_model, question)
    print(f"   {base_response}\n")
    
    # Fine-tuned model response
    print(f"🎯 FINE-TUNED MODEL:")
    finetuned_response = generate_response(finetuned_model, question)
    print(f"   {finetuned_response}\n")
    
    print(f"{'─'*60}\n")

print(f"\n{'='*60}")
print(f"  COMPARISON COMPLETED!")
print(f"{'='*60}\n")

print(f"📊 Analisi:")
print(f"   - Il modello BASE può dare risposte generiche o imprecise")
print(f"   - Il modello FINE-TUNED dovrebbe essere più specifico sui dati F1")
print(f"   - Cerca differenze in:")
print(f"     • Precisione dei dati (lap times, team, posizioni)")
print(f"     • Formato delle risposte")
print(f"     • Livello di dettaglio\n")

print(f"💾 Adapter salvato in: {ADAPTER_PATH}")
print(f"📁 Dimensione adapter: ~20-50MB (vs ~8GB modello base)\n")
