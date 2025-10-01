"""
Quick GPU Fine-tuning Test
Test rapido del fine-tuning con GPU senza dipendenze OllamaWrapper
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
print(f"  GPU FINE-TUNING TEST - F1 Dataset")
print(f"{'='*50}\n")

# Verifica GPU
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
print()

# Parametri
MODEL_NAME = "google/gemma-3-4b-it"
PROJECT_NAME = "f1_test_gpu"
EPOCHS = 1
BATCH_SIZE = 2
LIMIT = 20

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
# Usa il file più completo con Year, GrandPrix, EventDate
dataset = load_dataset("Vadera007/Formula_1_Dataset", data_files="f1_historical_data.csv", split="train")
print(f"✓ Dataset loaded: {len(dataset)} rows")
print()

# Prepare training data
print(f"📝 Preparing training data...")
training_data = []
for i, row in enumerate(dataset):
    if i >= LIMIT:
        break
    
    driver = row.get("Driver", "Unknown")
    team = row.get("Team", "Unknown")
    avg_lap = row.get("AvgLapTime")
    laps = row.get("LapsCompleted")
    
    # Skip if missing data
    if not avg_lap or not laps:
        continue
    
    prompt = f"Tell me about {driver}'s performance."
    response = f"{driver} from {team} completed {laps} laps with an average lap time of {avg_lap:.3f} seconds."
    
    training_data.append({
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
    })

print(f"✓ Created {len(training_data)} training examples")
print()

# Convert to Hugging Face dataset
from datasets import Dataset
train_dataset = Dataset.from_list(training_data)

# Load tokenizer
print(f"📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer loaded")
print()

# Tokenize function
def tokenize_function(examples):
    # Tokenize and set labels for causal language modeling
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    # For causal LM, labels are the same as input_ids (use list comprehension to avoid reference issues)
    result["labels"] = [input_ids[:] for input_ids in result["input_ids"]]
    return result

print(f"🔄 Tokenizing dataset...")
tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
print(f"✓ Dataset tokenized")
print()

# Data collator for causal LM
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)

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
print(f"✓ LoRA adapters added")
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
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator  # Use the correct data collator
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
print(f"  TEST COMPLETED SUCCESSFULLY!")
print(f"{'='*50}\n")
