"""
Test Inference - Fixed Version
Testa il modello fine-tuned con formato corretto
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

print(f"\n{'='*60}")
print(f"  INFERENCE TEST - FIXED VERSION")
print(f"{'='*60}\n")

# Load token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Parametri
MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_PATH = "./finetuning_projects/f1_expert_fixed/adapter"

# Check if adapter exists
if not os.path.exists(ADAPTER_PATH):
    print(f"❌ Adapter not found at: {ADAPTER_PATH}")
    print(f"\nEsegui prima il training:")
    print(f"  python test_training_fixed.py")
    exit(1)

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
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"✓ Tokenizer loaded")
print(f"  PAD token ID: {tokenizer.pad_token_id}")
print(f"  EOS token ID: {tokenizer.eos_token_id}")
print()

# Load base model
print(f"📥 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)
print(f"✓ Base model loaded\n")

# Load adapter
print(f"📥 Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()  # Modalità evaluation
print(f"✓ Adapter loaded\n")

def generate_response(prompt, max_new_tokens=150):
    """Generate response from fine-tuned model"""
    # Usa lo stesso formato del training
    formatted_prompt = f"Question: {prompt}\nAnswer: "
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decodifica solo i nuovi token
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()

# Test questions
TEST_QUESTIONS = [
    "Tell me about Lewis Hamilton's performance in F1.",
    "Who is Max Verstappen?",
    "What can you tell me about Mercedes F1 team?",
    "Tell me about Charles Leclerc.",
    "What do you know about Red Bull Racing?"
]

print(f"{'='*60}")
print(f"  TESTING FINE-TUNED MODEL")
print(f"{'='*60}\n")

for i, question in enumerate(TEST_QUESTIONS, 1):
    print(f"\n{'─'*60}")
    print(f"🔹 TEST {i}/{len(TEST_QUESTIONS)}")
    print(f"{'─'*60}")
    print(f"\n❓ {question}\n")
    
    try:
        response = generate_response(question)
        print(f"🎯 RISPOSTA:")
        print(f"   {response}\n")
    except Exception as e:
        print(f"❌ Errore: {e}\n")
    
    print(f"{'─'*60}\n")

print(f"\n{'='*60}")
print(f"  TEST COMPLETED!")
print(f"{'='*60}\n")

print(f"📊 Osservazioni:")
print(f"   - Il modello dovrebbe rispondere con dati specifici F1")
print(f"   - Cerca nomi di driver, team, lap times")
print(f"   - Confronta con le tue conoscenze F1")
print()

print(f"💾 Adapter location: {ADAPTER_PATH}")
print(f"📁 Adapter size: ~20-50MB")
print()
