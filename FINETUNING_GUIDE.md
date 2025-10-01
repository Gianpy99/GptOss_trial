# Fine-Tuning Guide for Ollama Wrapper

## 📚 Overview

This guide shows you how to fine-tune language models using your Ollama conversation history with **Hugging Face Transformers** and **PEFT (Parameter-Efficient Fine-Tuning)**.

## 🎯 Why Fine-Tune?

- **Specialize**: Adapt models to your specific domain or use case
- **Improve**: Teach models your preferred response style
- **Personalize**: Create assistants that understand your context
- **Efficient**: Use LoRA/QLoRA for memory-efficient training
- **Integrate**: Use existing conversations from OllamaWrapper as training data

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install base Ollama wrapper
pip install -r requirements.txt

# Install fine-tuning dependencies
pip install -r requirements-finetuning.txt
```

### 2. Create Training Data

Use the OllamaWrapper to create conversations:

```python
from ollama_wrapper import OllamaWrapper

wrapper = OllamaWrapper(session_id="training_python")
wrapper.chat("What are Python decorators?")
wrapper.chat("Show me an example")
# ... more conversations ...
```

### 3. Fine-Tune

```python
from ollama_wrapper import FineTuningManager

# Initialize manager
manager = FineTuningManager(
    model_name="microsoft/phi-2",
    use_4bit=True  # Use QLoRA for efficiency
)

# Load model and setup LoRA
manager.load_model()
manager.setup_lora(r=16, lora_alpha=32)

# Load training data from conversations
dataset = manager.load_training_data_from_memory()

# Tokenize
tokenized = manager.tokenize_dataset(dataset, max_length=512)

# Train
manager.train(
    train_dataset=tokenized,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    output_name="my_assistant"
)

# Save adapter (only a few MB!)
manager.save_adapter("my_assistant_adapter")
```

### 4. Use Fine-Tuned Model

```python
from ollama_wrapper import create_finetuned_assistant

assistant = create_finetuned_assistant(
    "./fine_tuned_models/my_assistant_adapter",
    temperature=0.7
)

response = assistant.generate_text(
    "User: Explain async/await\n\nAssistant:"
)
print(response)
```

## 📖 Detailed Usage

### Loading Training Data

**From Ollama Memory:**
```python
# All conversations
dataset = manager.load_training_data_from_memory()

# Specific sessions only
dataset = manager.load_training_data_from_memory(
    session_ids=["python_help", "coding_session"],
    min_length=20
)
```

**From JSON File:**
```python
# Format 1: Instruction-Output
data = [
    {"instruction": "What is X?", "output": "X is..."},
    {"instruction": "Explain Y", "output": "Y means..."}
]

# Format 2: Chat messages
data = [
    {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]}
]

# Load
dataset = manager.load_training_data_from_json("training.json")
```

### LoRA Configuration

**Conservative (Fast, Less Capacity):**
```python
manager.setup_lora(r=8, lora_alpha=16)
```

**Balanced (Recommended):**
```python
manager.setup_lora(r=16, lora_alpha=32)
```

**Aggressive (Slower, More Capacity):**
```python
manager.setup_lora(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"]
)
```

### Training Parameters

```python
manager.train(
    train_dataset=train_data,
    eval_dataset=eval_data,        # Optional
    num_epochs=3,                   # Training epochs
    batch_size=4,                   # Samples per batch
    learning_rate=2e-4,             # Learning rate
    gradient_accumulation_steps=4,  # Effective batch = batch_size * this
    warmup_steps=100,               # LR warmup
    save_steps=100,                 # Save checkpoint frequency
    output_name="my_model"
)
```

### Generation Options

```python
response = assistant.generate_text(
    prompt="User: Hello\n\nAssistant:",
    max_new_tokens=256,    # Max tokens to generate
    temperature=0.7,       # Randomness (0=deterministic, 1+=creative)
    top_p=0.9,            # Nucleus sampling
    top_k=50,             # Top-k sampling
    do_sample=True        # Use sampling vs greedy
)
```

## 🔧 Advanced Features

### Custom Target Modules

Different models have different architectures. Specify which layers to adapt:

```python
# For Llama/Mistral-style models
manager.setup_lora(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# For GPT-style models
manager.setup_lora(
    target_modules=["c_attn", "c_proj"]
)

# More aggressive (slower but more capacity)
manager.setup_lora(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
```

### Data Splitting

```python
from datasets import DatasetDict

dataset = manager.load_training_data_from_memory()

# Split 85% train, 15% eval
split = dataset.train_test_split(test_size=0.15, seed=42)

train_tokenized = manager.tokenize_dataset(split["train"])
eval_tokenized = manager.tokenize_dataset(split["test"])

manager.train(
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized
)
```

### Multiple Adapters

Create specialized adapters for different tasks:

```python
# Train Python expert
manager.setup_lora(r=16, lora_alpha=32)
python_data = manager.load_training_data_from_memory(
    session_ids=["python_sessions"]
)
tokenized = manager.tokenize_dataset(python_data)
manager.train(tokenized, output_name="python_expert")
manager.save_adapter("python_expert_adapter")

# Train JavaScript expert (reload base model first)
manager = FineTuningManager(model_name="microsoft/phi-2")
manager.load_model()
manager.setup_lora(r=16, lora_alpha=32)
js_data = manager.load_training_data_from_memory(
    session_ids=["javascript_sessions"]
)
tokenized = manager.tokenize_dataset(js_data)
manager.train(tokenized, output_name="javascript_expert")
manager.save_adapter("javascript_expert_adapter")

# Use them
python_assistant = create_finetuned_assistant("./fine_tuned_models/python_expert_adapter")
js_assistant = create_finetuned_assistant("./fine_tuned_models/javascript_expert_adapter")
```

### Hybrid Workflow

Combine Ollama (fast inference) with fine-tuned models (specialized):

```python
from ollama_wrapper import OllamaWrapper, create_finetuned_assistant

# General purpose (fast, local Ollama)
ollama = OllamaWrapper(model="gemma3:4b")

# Specialized (fine-tuned)
specialist = create_finetuned_assistant("./my_adapter")

def smart_chat(user_query):
    # Route to specialist for specific topics
    if any(keyword in user_query.lower() for keyword in ["python", "code", "programming"]):
        prompt = f"User: {user_query}\n\nAssistant:"
        return specialist.generate_text(prompt)
    else:
        # Use Ollama for general queries
        return ollama.chat(user_query)

# Usage
response = smart_chat("How do I use Python decorators?")  # -> Specialist
response = smart_chat("What's the weather like?")         # -> Ollama
```

## 💡 Best Practices

### 1. Data Quality Over Quantity
- **Better**: 50 high-quality, diverse conversations
- **Worse**: 500 repetitive, low-quality exchanges

### 2. Start Small
- Use smaller models for testing (phi-2, llama-7b)
- Scale up only when needed

### 3. Use QLoRA (4-bit)
```python
manager = FineTuningManager(use_4bit=True)  # Much more memory efficient
```

### 4. Monitor Evaluation Metrics
Always use an evaluation set to prevent overfitting:
```python
split = dataset.train_test_split(test_size=0.15)
manager.train(
    train_dataset=split["train"],
    eval_dataset=split["test"]  # ← Important!
)
```

### 5. Experiment with LoRA Rank
- **r=8**: Fast, minimal overhead, good for simple adaptations
- **r=16**: Balanced (recommended starting point)
- **r=32+**: More capacity, slower, for complex tasks

### 6. Save Checkpoints
```python
manager.train(
    ...,
    save_steps=100,  # Save every 100 steps
    save_total_limit=3  # Keep only 3 most recent checkpoints
)
```

### 7. Format Consistency
Use consistent prompt formatting:
```python
# Good: Consistent format
"User: {question}\n\nAssistant: {answer}"

# Bad: Inconsistent
"Q: {question} A: {answer}"  # Mixed with
"User: {question}\nBot: {answer}"
```

## 📊 Expected Results

### Model Sizes
- **Base Model**: 3-13 GB (depending on model)
- **LoRA Adapter**: 2-50 MB (much smaller!)
- **Full Fine-tuned**: Same as base model

### Training Time (rough estimates)
- **100 examples, r=16, 3 epochs**:
  - GPU (RTX 3090): ~5-15 minutes
  - GPU (GTX 1080): ~15-30 minutes
  - CPU: Not recommended (hours/days)

### Memory Requirements
- **QLoRA (4-bit)**: ~6-8 GB GPU RAM for 7B model
- **Full precision**: ~14-20 GB GPU RAM for 7B model

## 🐛 Troubleshooting

### Out of Memory
```python
# Solutions:
manager = FineTuningManager(use_4bit=True)  # Use QLoRA
manager.train(
    batch_size=1,              # Reduce batch size
    gradient_accumulation_steps=16  # Increase accumulation
)
```

### Dependencies Not Found
```bash
# Install all fine-tuning dependencies
pip install -r requirements-finetuning.txt

# Or individually:
pip install transformers peft torch datasets accelerate bitsandbytes
```

### CUDA Not Available
```python
# Check
import torch
print(torch.cuda.is_available())

# If False, install CUDA-enabled PyTorch:
# Visit: https://pytorch.org/get-started/locally/
```

### Slow Training
- Use GPU instead of CPU
- Enable 4-bit quantization (`use_4bit=True`)
- Reduce `batch_size`, increase `gradient_accumulation_steps`
- Use smaller `max_length` in tokenization

### Model Quality Issues
- Need more diverse training data
- Try higher LoRA rank (r=32)
- Train for more epochs
- Check evaluation loss (should decrease)

## 📚 Examples

See `examples/example_finetuning_integration.py` for complete examples:

```bash
# Interactive menu
python examples/example_finetuning_integration.py

# Run specific steps
python examples/example_finetuning_integration.py step1  # Create data
python examples/example_finetuning_integration.py step2  # Fine-tune
python examples/example_finetuning_integration.py step3  # Test
python examples/example_finetuning_integration.py full   # Full pipeline
```

## 🔗 Resources

- **PEFT Documentation**: https://huggingface.co/docs/peft
- **Transformers Guide**: https://huggingface.co/docs/transformers
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314

## 🤝 Contributing

Have ideas for improving the fine-tuning workflow? Open an issue or PR!

## 📄 License

Same as the main Ollama_wrapper project.
