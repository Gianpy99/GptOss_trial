"""
Training Hybrid Expert (F1 + Tolkien) - Ottimizzato per GTX 1660 SUPER
"""

import os
import sys
import io
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Forza UTF-8 su Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print(f"\n{'='*60}")
print(f"  🔮 HYBRID EXPERT TRAINING (F1 + Tolkien)")
print(f"  Ottimizzato per GTX 1660 SUPER (6GB)")
print(f"{'='*60}\n")

# Verifica GPU
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
else:
    print("❌ GPU non disponibile")
    sys.exit(1)
print()

# Parametri
MODEL_NAME = "google/gemma-3-4b-it"
PROJECT_NAME = "hybrid_expert"
DATASET_FILE = "combined_f1_tolkien_data.json"
OUTPUT_DIR = f"finetuning_projects/{PROJECT_NAME}"
EPOCHS = 2
BATCH_SIZE = 2
LIMIT = 100  # Usa 100 esempi (mix F1 + Tolkien)
MAX_LENGTH = 512

print(f"⚙️ Parametri:")
print(f"  Model: {MODEL_NAME}")
print(f"  Project: {PROJECT_NAME}")
print(f"  Dataset: {DATASET_FILE}")
print(f"  Examples: {LIMIT}")
print(f"  Epochs: {EPOCHS}")
print(f"  Output: {OUTPUT_DIR}")
print()

# Carica dataset
print(f"📥 Loading dataset...")
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prendi i primi LIMIT esempi
if len(data) > LIMIT:
    data = data[:LIMIT]

print(f"✓ Loaded {len(data)} examples")
print(f"  - Sample instruction: {data[0].get('instruction', '')[:50]}...")
print()

# Converti in Dataset
def format_example(example):
    instruction = example.get('instruction', '')
    output = example.get('output') or example.get('response') or example.get('completion', '')
    return {
        "text": f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    }

formatted_data = [format_example(ex) for ex in data]
dataset = Dataset.from_list(formatted_data)

print(f"📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"✓ Tokenizer loaded")
print()

# Tokenizza dataset
print(f"🔄 Tokenizing (max_length={MAX_LENGTH})...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)
print(f"✓ Tokenization complete")
print()

# 4-bit quantization config
print(f"⚙️ Setting up 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print(f"✓ Config ready")
print()

# Carica modello
print(f"📥 Loading model (2-3 minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
print(f"✓ Model loaded")
print()

# Prepara modello per training
print(f"⚙️ Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)
print(f"✓ Model prepared")
print()

# LoRA config
print(f"⚙️ Setting up LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"✓ LoRA configured")
print(f"  Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
print()

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    optim="paged_adamw_8bit",
    warmup_steps=10,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.3
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
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Training
print(f"{'='*60}")
print(f"  🚀 STARTING TRAINING")
print(f"  Expected time: ~15-20 minutes")
print(f"{'='*60}\n")

try:
    trainer.train()
    print(f"\n✅ Training completato!")
    
    # Salva l'adapter finale
    print(f"\n💾 Saving adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Saved to {OUTPUT_DIR}")
    
    print(f"\n{'='*60}")
    print(f"  ✅ HYBRID EXPERT TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📍 Next steps:")
    print(f"  1. Merge: python ollama_cli.py merge --project {PROJECT_NAME}")
    print(f"  2. Deploy: python ollama_cli.py deploy --project {PROJECT_NAME}")
    print(f"  3. Test: ollama run {PROJECT_NAME}")
    print()

except KeyboardInterrupt:
    print(f"\n\n⚠️  Training interrotto dall'utente")
    print(f"📍 Checkpoint salvato in: {OUTPUT_DIR}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Errore durante training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
