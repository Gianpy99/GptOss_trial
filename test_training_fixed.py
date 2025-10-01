"""
Fixed GPU Fine-tuning - Correct Tokenization
Usa il formato corretto senza token speciali inventati
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print(f"\n{'='*50}")
print(f"  GPU FINE-TUNING - FIXED VERSION")
print(f"{'='*50}\n")

# Verifica GPU
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
print()

# Parametri
MODEL_NAME = "google/gemma-3-4b-it"
PROJECT_NAME = "f1_expert_fixed"
EPOCHS = 3
BATCH_SIZE = 2
LIMIT = 50  # Più esempi per risultati migliori

print(f"Parametri:")
print(f"  Model: {MODEL_NAME}")
print(f"  Project: {PROJECT_NAME}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Training Examples: {LIMIT}")
print()

# Load token
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    print("✓ HF Token loaded")
else:
    print("⚠ HF Token not found")
print()

# Download dataset
print(f"📥 Downloading F1 dataset...")
dataset = load_dataset("Vadera007/Formula_1_Dataset", data_files="f1_historical_data.csv", split="train")
print(f"✓ Dataset loaded: {len(dataset)} rows")
print()

# Prepare training data - FORMATO SEMPLICE SENZA TOKEN SPECIALI
print(f"📝 Preparing training data...")
training_texts = []
for i, row in enumerate(dataset):
    if i >= LIMIT:
        break
    
    driver = row.get("Driver", "Unknown")
    team = row.get("Team", "Unknown")
    avg_lap = row.get("AvgLapTime")
    laps = row.get("LapsCompleted")
    quali = row.get("QualiPosition")
    finish = row.get("RaceFinishPosition")
    year = row.get("Year", "")
    gp = row.get("GrandPrix", "")
    
    # Skip if missing critical data
    if not avg_lap or not laps:
        continue
    
    # Formato semplice Q&A senza token speciali
    prompt = f"Question: Tell me about {driver}'s performance in F1.\nAnswer: "
    
    response = f"{driver} from {team} completed {laps} laps with an average lap time of {avg_lap:.3f} seconds"
    
    if quali:
        response += f", starting from position {int(quali)}"
    if finish:
        response += f" and finishing in position {int(finish)}"
    if year and gp:
        response += f" at the {gp} Grand Prix {int(year)}"
    response += "."
    
    # Formato completo per training
    full_text = prompt + response
    training_texts.append(full_text)

print(f"✓ Created {len(training_texts)} training examples")
print()

# Convert to HF dataset
from datasets import Dataset
train_dataset = Dataset.from_dict({"text": training_texts})

# Load tokenizer
print(f"📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# Setup pad token properly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"✓ Tokenizer loaded")
print(f"  Vocab size: {len(tokenizer)}")
print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print()

# Tokenize function - CORRETTO con padding
def tokenize_function(examples):
    # Tokenizza il testo con padding per uniformare le lunghezze
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"  # Padding a lunghezza fissa
    )
    # Per Causal LM: labels = input_ids
    result["labels"] = result["input_ids"].copy()
    return result

print(f"🔄 Tokenizing dataset...")
tokenized_dataset = train_dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=["text"]
)
print(f"✓ Dataset tokenized")
print()

# Non serve più il data collator con padding="max_length"
data_collator = None

# Quantization config
print(f"⚙️ Setting up 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print(f"✓ Quantization config ready")
print()

# Load model
print(f"📥 Loading model (this may take 2-3 minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)
print(f"✓ Model loaded to GPU")
print()

# Prepare for training
print(f"🔧 Preparing model for training...")
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ LoRA adapters added")
print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print()

# Training arguments
output_dir = f"./finetuning_projects/{PROJECT_NAME}"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    logging_steps=5,
    save_steps=50,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
    # data_collator non necessario con padding="max_length"
)

print(f"{'='*50}")
print(f"  STARTING TRAINING")
print(f"{'='*50}\n")

import time
start_time = time.time()

# Train
trainer.train()

elapsed_time = time.time() - start_time
print(f"\n{'='*50}")
print(f"  TRAINING COMPLETED!")
print(f"{'='*50}")
print(f"\n⏱️ Training time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
print(f"✓ Model saved to: {output_dir}")
print()

# Save adapter
adapter_dir = f"{output_dir}/adapter"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"✓ LoRA adapter saved to: {adapter_dir}")
print()

print(f"{'='*50}")
print(f"  SUCCESS!")
print(f"{'='*50}\n")
print(f"Prossimi passi:")
print(f"1. Test inference: python test_inference_fixed.py")
print(f"2. Deploy: python finetuning_workflow.py deploy --project {PROJECT_NAME}")
print()
