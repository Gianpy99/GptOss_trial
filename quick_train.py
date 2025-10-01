"""
Training rapido con dataset personalizzato
Usa: python quick_train.py <dataset.json> <project_name> [num_examples]
"""
import sys
import json
import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Parametri da command line
DATASET_FILE = sys.argv[1] if len(sys.argv) > 1 else "combined_f1_tolkien_data.json"
PROJECT_NAME = sys.argv[2] if len(sys.argv) > 2 else "hybrid_expert"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 50  # Default 50 esempi

# Configurazione
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = f"finetuning_projects/{PROJECT_NAME}"
EPOCHS = 2
BATCH_SIZE = 2
MAX_LENGTH = 512

print(f"\n{'='*60}")
print(f"  🚀 QUICK TRAINING")
print(f"{'='*60}\n")
print(f"📊 Dataset: {DATASET_FILE}")
print(f"📁 Project: {PROJECT_NAME}")
print(f"📈 Examples: {LIMIT}")
print(f"⚙️  Epochs: {EPOCHS}")
print(f"💾 Output: {OUTPUT_DIR}")
print()

# Carica dataset
print(f"📥 Loading dataset...")
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Normalizza
examples = []
for item in data[:LIMIT]:
    if isinstance(item, dict):
        instr = item.get('instruction') or item.get('prompt', '')
        resp = item.get('response') or item.get('completion') or item.get('output', '')
        if instr and resp:
            text = f"<start_of_turn>user\n{instr}<end_of_turn>\n<start_of_turn>model\n{resp}<end_of_turn>"
            examples.append({'text': text})

print(f"✅ Loaded {len(examples)} training examples")
print()

# Load HF token
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Load tokenizer
print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded")
print()

# Tokenize
print("🔄 Tokenizing...")
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

dataset = Dataset.from_list(examples)
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
print("✅ Tokenization complete")
print()

# Quantization config
print("⚙️ Setting up 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
print("✅ Config ready")
print()

# Load model
print("📥 Loading model (2-3 minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True
)
print("✅ Model loaded to GPU")
print()

# Prepare for training
print("🔧 Preparing model...")
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,  # Rank più alto per dataset combinato
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"✅ LoRA adapters added")
print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print()

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    warmup_steps=10,
    report_to="none"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

# Train!
print(f"{'='*60}")
print(f"  🚀 STARTING TRAINING")
print(f"  Estimated time: ~{EPOCHS * len(examples) // 2} minutes")
print(f"{'='*60}\n")

trainer.train()

print(f"\n{'='*60}")
print(f"  ✅ TRAINING COMPLETE!")
print(f"{'='*60}\n")

# Save adapter
adapter_dir = Path(OUTPUT_DIR) / "adapter"
adapter_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

print(f"💾 Adapter saved to: {adapter_dir}")
print(f"\n🎯 Next steps:")
print(f"1. Merge adapter: python merge_and_deploy.py {PROJECT_NAME}")
print(f"2. Or import directly to Ollama")
