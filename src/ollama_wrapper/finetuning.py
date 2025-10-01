"""
Fine-tuning utilities using Hugging Face Transformers and PEFT.
Integrates seamlessly with the OllamaWrapper memory system.

This module allows you to:
1. Fine-tune models using conversation history from MemoryManager
2. Apply PEFT techniques (LoRA, QLoRA) for efficient training
3. Export fine-tuned models back to Ollama format (GGUF)
4. Create adapters that can be loaded into your workflow

Example:
    from ollama_wrapper import OllamaWrapper, FineTuningManager
    
    # Create some training conversations
    wrapper = OllamaWrapper()
    wrapper.chat("Explain Python decorators")
    
    # Fine-tune using those conversations
    ft_manager = FineTuningManager()
    ft_manager.load_model("microsoft/phi-2")
    ft_manager.setup_lora()
    
    dataset = ft_manager.load_training_data_from_memory()
    tokenized = ft_manager.tokenize_dataset(dataset)
    ft_manager.train(tokenized)
"""

import os
import json
import sqlite3
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path

# Optional imports - only fail if actually used
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
        PeftModel,
    )
    from datasets import Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    # Create dummy for runtime when not installed
    torch = None
    Dataset = None
    if TYPE_CHECKING:
        # Only for type checkers - will never execute
        from datasets import Dataset

from .wrapper import MEMORY_DB


class FineTuningManager:
    """
    Manages fine-tuning workflows using PEFT and Hugging Face.
    Integrates with OllamaWrapper's MemoryManager for training data.
    
    Attributes:
        model_name: Hugging Face model identifier
        output_dir: Directory to save fine-tuned models
        memory_db_path: Path to SQLite memory database
        use_4bit: Use 4-bit quantization (QLoRA) for memory efficiency
        model: The loaded model (None until load_model() is called)
        tokenizer: The loaded tokenizer
        peft_config: LoRA configuration
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        output_dir: str = "./fine_tuned_models",
        memory_db_path: str = MEMORY_DB,
        use_4bit: bool = True,
    ):
        """
        Initialize the fine-tuning manager.

        Args:
            model_name: Hugging Face model identifier (e.g., "microsoft/phi-2")
            output_dir: Directory to save fine-tuned models and adapters
            memory_db_path: Path to SQLite memory database (uses OllamaWrapper default)
            use_4bit: Use 4-bit quantization (QLoRA) for memory efficiency

        Raises:
            ImportError: If required dependencies are not installed
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "Fine-tuning dependencies not installed. Install with:\n"
                "pip install transformers peft torch datasets accelerate bitsandbytes"
            )

        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_db_path = memory_db_path
        self.use_4bit = use_4bit

        self.model = None
        self.tokenizer = None
        self.peft_config = None

    def load_model(
        self,
        load_in_4bit: Optional[bool] = None,
        device_map: str = "auto",
        torch_dtype: Optional[Any] = None,
    ):
        """
        Load the base model and tokenizer from Hugging Face.

        Args:
            load_in_4bit: Override the default 4-bit loading setting
            device_map: Device mapping strategy ("auto", "cuda", "cpu")
            torch_dtype: Override torch data type (default: float16)

        Example:
            manager.load_model()  # Uses default 4-bit quantization
            manager.load_model(load_in_4bit=False)  # Full precision
        """
        if load_in_4bit is None:
            load_in_4bit = self.use_4bit

        print(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization if using 4-bit
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Determine torch dtype
        if torch_dtype is None:
            torch_dtype = torch.float16 if not load_in_4bit else None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

        # Prepare for k-bit training if quantized
        if load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        print(f"✓ Model loaded successfully. Device: {self.model.device}")

    def setup_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
    ):
        """
        Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning.
        
        LoRA adds trainable rank decomposition matrices to existing weights,
        dramatically reducing the number of trainable parameters.

        Args:
            r: LoRA attention dimension (rank). Higher = more capacity but slower
            lora_alpha: LoRA scaling factor. Typically 2*r
            lora_dropout: Dropout probability for LoRA layers
            target_modules: Modules to apply LoRA to (auto-detected if None)
            bias: Bias training strategy ("none", "all", "lora_only")

        Example:
            # Conservative (fast, less capacity)
            manager.setup_lora(r=8, lora_alpha=16)
            
            # Aggressive (slower, more capacity)
            manager.setup_lora(r=32, lora_alpha=64, target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"
            ])
        """
        if self.model is None:
            raise ValueError("Model must be loaded first. Call load_model().")

        # Auto-detect target modules if not provided
        if target_modules is None:
            # Common patterns for different architectures
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        self.peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, self.peft_config)
        print("✓ LoRA configuration applied:")
        self.model.print_trainable_parameters()

    def load_training_data_from_memory(
        self,
        session_ids: Optional[List[str]] = None,
        min_length: int = 10,
        format_style: str = "chat",
    ) -> "Dataset":
        """
        Load training data from the OllamaWrapper MemoryManager SQLite database.
        
        This integrates seamlessly with your existing conversation history,
        allowing you to fine-tune on real interactions.

        Args:
            session_ids: Filter by specific session IDs (all sessions if None)
            min_length: Minimum message content length to include
            format_style: How to format conversations ("chat", "instruct", "completion")

        Returns:
            Hugging Face Dataset ready for training

        Example:
            # Use all conversations
            dataset = manager.load_training_data_from_memory()
            
            # Use specific sessions only
            dataset = manager.load_training_data_from_memory(
                session_ids=["coding_session", "python_help"]
            )
        """
        if not os.path.exists(self.memory_db_path):
            print(f"⚠ Memory database not found: {self.memory_db_path}")
            return Dataset.from_list([])

        conn = sqlite3.connect(self.memory_db_path)
        cursor = conn.cursor()

        # Build query - check for both 'conversations' and 'messages' tables
        # (supporting both old and new schema)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('conversations', 'messages')"
        )
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            print("⚠ No conversation tables found in database")
            conn.close()
            return Dataset.from_list([])

        table_name = tables[0]  # Use whichever exists

        query = f"""
            SELECT role, content, session_id, timestamp
            FROM {table_name}
            WHERE length(content) >= ?
        """
        params = [min_length]

        if session_ids:
            placeholders = ",".join("?" * len(session_ids))
            query += f" AND session_id IN ({placeholders})"
            params.extend(session_ids)

        query += " ORDER BY timestamp"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # Convert to conversation format
        conversations = []
        current_conversation = []
        current_session = None

        for role, content, session_id, timestamp in rows:
            if current_session != session_id and current_conversation:
                # New session - save previous conversation
                text = self._format_conversation(current_conversation, format_style)
                if text:
                    conversations.append({
                        "text": text,
                        "session_id": current_session,
                        "num_turns": len(current_conversation)
                    })
                current_conversation = []

            current_session = session_id
            current_conversation.append({"role": role, "content": content})

            # Create training example when we have a complete exchange
            if role == "assistant" and len(current_conversation) >= 2:
                text = self._format_conversation(current_conversation, format_style)
                if text:
                    conversations.append({
                        "text": text,
                        "session_id": session_id,
                        "num_turns": len(current_conversation)
                    })
                current_conversation = []

        # Handle remaining conversation
        if current_conversation:
            text = self._format_conversation(current_conversation, format_style)
            if text:
                conversations.append({
                    "text": text,
                    "session_id": current_session,
                    "num_turns": len(current_conversation)
                })

        print(f"✓ Loaded {len(conversations)} conversation examples from memory")
        if conversations:
            avg_turns = sum(c["num_turns"] for c in conversations) / len(conversations)
            print(f"  Average turns per example: {avg_turns:.1f}")

        return Dataset.from_list(conversations)

    def load_training_data_from_json(
        self,
        json_path: str,
        format_style: str = "auto",
    ) -> "Dataset":
        """
        Load training data from a JSON file.

        Supports multiple formats:
        - [{"instruction": "...", "output": "..."}]
        - [{"prompt": "...", "completion": "..."}]
        - [{"text": "..."}]
        - [{"messages": [{"role": "user", "content": "..."}, ...]}]

        Args:
            json_path: Path to JSON file
            format_style: How to format data ("auto", "chat", "instruct", "completion")

        Returns:
            Hugging Face Dataset

        Example:
            dataset = manager.load_training_data_from_json("my_data.json")
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalize to text format
        normalized = []
        for item in data:
            if "text" in item:
                normalized.append({"text": item["text"]})
            elif "messages" in item:
                text = self._format_conversation(item["messages"], format_style)
                if text:
                    normalized.append({"text": text})
            elif "instruction" in item and "output" in item:
                if format_style == "auto" or format_style == "instruct":
                    text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
                else:
                    text = f"User: {item['instruction']}\n\nAssistant: {item['output']}"
                normalized.append({"text": text})
            elif "prompt" in item and "completion" in item:
                text = f"{item['prompt']}\n{item['completion']}"
                normalized.append({"text": text})

        print(f"✓ Loaded {len(normalized)} examples from {json_path}")
        return Dataset.from_list(normalized)

    def _format_conversation(
        self,
        messages: List[Dict[str, str]],
        style: str = "chat",
    ) -> str:
        """
        Format a conversation into a single training text.

        Args:
            messages: List of message dicts with 'role' and 'content'
            style: Formatting style ("chat", "instruct", "completion")

        Returns:
            Formatted text string
        """
        if not messages:
            return ""

        if style == "chat":
            # Chat format: "User: ... \n\nAssistant: ..."
            formatted_parts = []
            for msg in messages:
                role = msg["role"].capitalize()
                if role == "System":
                    formatted_parts.append(f"System: {msg['content']}")
                elif role == "User":
                    formatted_parts.append(f"User: {msg['content']}")
                elif role == "Assistant":
                    formatted_parts.append(f"Assistant: {msg['content']}")
            return "\n\n".join(formatted_parts)

        elif style == "instruct":
            # Instruction format
            if len(messages) >= 2:
                instruction = messages[0]["content"]
                response = messages[-1]["content"]
                return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            return ""

        elif style == "completion":
            # Simple completion format
            return " ".join(msg["content"] for msg in messages)

        else:
            # Default to chat
            return self._format_conversation(messages, "chat")

    def tokenize_dataset(
        self,
        dataset: "Dataset",
        max_length: int = 512,
        remove_columns: Optional[List[str]] = None,
    ) -> "Dataset":
        """
        Tokenize a dataset for training.

        Args:
            dataset: Input dataset with 'text' field
            max_length: Maximum sequence length (longer sequences are truncated)
            remove_columns: Columns to remove (auto-detected if None)

        Returns:
            Tokenized dataset ready for training

        Example:
            dataset = manager.load_training_data_from_memory()
            tokenized = manager.tokenize_dataset(dataset, max_length=512)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        if remove_columns is None:
            remove_columns = dataset.column_names

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_columns,
            desc="Tokenizing dataset",
        )

        print(f"✓ Dataset tokenized: {len(tokenized)} examples")
        return tokenized

    def train(
        self,
        train_dataset: "Dataset",
        eval_dataset: Optional["Dataset"] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        save_steps: int = 100,
        logging_steps: int = 10,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        output_name: str = "fine_tuned_model",
    ):
        """
        Fine-tune the model using the prepared dataset.

        Args:
            train_dataset: Tokenized training dataset
            eval_dataset: Optional tokenized evaluation dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size per device
            learning_rate: Learning rate for optimizer
            save_steps: Save checkpoint every N steps
            logging_steps: Log metrics every N steps
            gradient_accumulation_steps: Accumulate gradients over N steps
            warmup_steps: Number of warmup steps for learning rate scheduler
            output_name: Name for the output directory

        Example:
            manager.train(
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                num_epochs=3,
                batch_size=4,
                learning_rate=2e-4,
                output_name="my_model"
            )
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        output_path = self.output_dir / output_name

        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none",  # Disable wandb, tensorboard etc.
            fp16=True if torch.cuda.is_available() else False,
            push_to_hub=False,
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        print(f"\n{'='*60}")
        print(f"Starting training:")
        print(f"  - Model: {self.model_name}")
        print(f"  - Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"  - Evaluation samples: {len(eval_dataset)}")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Output: {output_path}")
        print(f"{'='*60}\n")

        trainer.train()

        print(f"\n✓ Training complete! Model saved to {output_path}")

    def save_adapter(self, adapter_name: str):
        """
        Save only the LoRA adapter weights (much smaller than full model).
        
        Adapter files are typically only a few MB instead of several GB,
        making them easy to share and version control.

        Args:
            adapter_name: Name for the adapter directory

        Example:
            manager.save_adapter("my_coding_assistant_adapter")
        """
        if self.model is None:
            raise ValueError("No model to save.")

        adapter_path = self.output_dir / adapter_name
        self.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        
        # Calculate size
        total_size = sum(
            f.stat().st_size for f in adapter_path.rglob('*') if f.is_file()
        )
        size_mb = total_size / (1024 * 1024)
        
        print(f"✓ Adapter saved to {adapter_path}")
        print(f"  Size: {size_mb:.1f} MB")

    def load_adapter(self, adapter_path: str):
        """
        Load a previously saved LoRA adapter.

        Args:
            adapter_path: Path to the adapter directory

        Example:
            manager.load_adapter("./fine_tuned_models/my_adapter")
        """
        if self.model is None:
            self.load_model()

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        print(f"✓ Adapter loaded from {adapter_path}")

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = 50,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text using the fine-tuned model.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Use sampling (True) or greedy decoding (False)

        Returns:
            Generated text string

        Example:
            prompt = "User: Explain Python decorators\\n\\nAssistant:"
            response = manager.generate_text(prompt, max_new_tokens=150)
            print(response)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    def export_to_gguf(
        self,
        output_path: str,
        quantization: str = "q4_k_m",
    ):
        """
        Export the fine-tuned model to GGUF format for use with Ollama.
        
        NOTE: This requires llama.cpp to be installed separately.
        See: https://github.com/ggerganov/llama.cpp

        Args:
            output_path: Path for output GGUF file
            quantization: Quantization type (q4_k_m, q5_k_m, q8_0, etc.)

        Example:
            manager.export_to_gguf(
                "./my_model.gguf",
                quantization="q4_k_m"
            )
        """
        raise NotImplementedError(
            "GGUF export requires llama.cpp integration. "
            "This is planned for a future version. "
            "For now, use the model directly in Python or "
            "manually convert using llama.cpp tools."
        )


def create_finetuned_assistant(
    adapter_path: str,
    base_model: str = "microsoft/phi-2",
    **generate_kwargs
):
    """
    Factory function to quickly create a fine-tuned assistant.
    
    Args:
        adapter_path: Path to the saved LoRA adapter
        base_model: Base model name
        **generate_kwargs: Default generation parameters
        
    Returns:
        FineTuningManager instance with loaded adapter
        
    Example:
        assistant = create_finetuned_assistant(
            "./fine_tuned_models/coding_assistant",
            temperature=0.2,
            max_new_tokens=200
        )
        
        response = assistant.generate_text("User: How do I use async/await?\\n\\nAssistant:")
    """
    manager = FineTuningManager(model_name=base_model)
    manager.load_adapter(adapter_path)
    
    # Store default generation kwargs
    manager._default_generate_kwargs = generate_kwargs
    
    # Monkey-patch generate to use defaults
    original_generate = manager.generate_text
    def generate_with_defaults(prompt: str, **kwargs):
        merged_kwargs = {**manager._default_generate_kwargs, **kwargs}
        return original_generate(prompt, **merged_kwargs)
    manager.generate_text = generate_with_defaults
    
    return manager


# Example usage
if __name__ == "__main__":
    print("Fine-tuning module loaded successfully!")
    print("\nTo use fine-tuning, install dependencies:")
    print("  pip install transformers peft torch datasets accelerate bitsandbytes")
    print("\nExample usage:")
    print("  from ollama_wrapper import FineTuningManager")
    print("  manager = FineTuningManager()")
    print("  manager.load_model()")
    print("  manager.setup_lora()")
    print("  dataset = manager.load_training_data_from_memory()")
    print("  tokenized = manager.tokenize_dataset(dataset)")
    print("  manager.train(tokenized)")
