"""
Complete example of integrating OllamaWrapper with FineTuningManager.

This demonstrates the full workflow:
1. Create training conversations using OllamaWrapper
2. Fine-tune a model using those conversations
3. Test the fine-tuned model
4. Integrate it back into your workflow

Requirements:
    pip install -r requirements.txt
    pip install -r requirements-finetuning.txt
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ollama_wrapper import (
    OllamaWrapper,
    MemoryManager,
)

try:
    from ollama_wrapper import FineTuningManager, create_finetuned_assistant
    FINETUNING_AVAILABLE = True
except ImportError:
    FINETUNING_AVAILABLE = False
    print("⚠ Fine-tuning not available. Install dependencies:")
    print("  pip install -r requirements-finetuning.txt")


def step1_create_training_data():
    """
    Step 1: Create training conversations using OllamaWrapper.
    
    This simulates collecting real user interactions that you want
    your fine-tuned model to learn from.
    """
    print("\n" + "="*70)
    print("STEP 1: Creating Training Data with OllamaWrapper")
    print("="*70 + "\n")

    wrapper = OllamaWrapper(model="gemma3:4b")

    # Create specialized conversations about Python programming
    python_topics = [
        ("decorators", [
            "What are Python decorators?",
            "Show me an example of a simple decorator",
            "How do I pass arguments to a decorator?",
        ]),
        ("async", [
            "Explain async/await in Python",
            "When should I use asyncio?",
            "Show me an example of async function",
        ]),
        ("typing", [
            "What are type hints in Python?",
            "How do I use Optional and Union types?",
            "What is the difference between List and list in type hints?",
        ]),
    ]

    for topic, questions in python_topics:
        session_id = f"python_{topic}_training"
        wrapper.session_id = session_id
        
        print(f"\n📚 Creating training session: {topic}")
        for question in questions:
            print(f"  Q: {question[:50]}...")
            try:
                response = wrapper.chat(question)
                print(f"  A: {response[:80]}...")
            except Exception as e:
                print(f"  ⚠ Error: {e}")

    print(f"\n✓ Training data created in memory database")
    
    # Show what we have
    memory = MemoryManager()
    conn = memory._conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversations")
    count = cursor.fetchone()[0]
    conn.close()
    
    print(f"  Total messages in database: {count}")


def step2_finetune_model():
    """
    Step 2: Fine-tune a model using the collected conversations.
    """
    if not FINETUNING_AVAILABLE:
        print("\n⚠ Skipping fine-tuning - dependencies not installed")
        return

    print("\n" + "="*70)
    print("STEP 2: Fine-Tuning Model with Collected Data")
    print("="*70 + "\n")

    # Initialize fine-tuning manager
    manager = FineTuningManager(
        model_name="microsoft/phi-2",  # Small, efficient model
        output_dir="./fine_tuned_models",
        use_4bit=True,  # Use QLoRA for memory efficiency
    )

    print("Loading base model...")
    manager.load_model()

    print("\nConfiguring LoRA...")
    manager.setup_lora(
        r=16,           # LoRA rank
        lora_alpha=32,  # LoRA alpha (scaling)
        lora_dropout=0.05,
    )

    print("\nLoading training data from memory...")
    dataset = manager.load_training_data_from_memory(
        session_ids=None,  # Use all sessions
        min_length=10,
        format_style="chat",
    )

    if len(dataset) < 5:
        print("⚠ Not enough training data. Need at least 5 examples.")
        print("  Run step1_create_training_data() first or create more conversations.")
        return

    print(f"\nDataset statistics:")
    print(f"  Total examples: {len(dataset)}")
    
    # Split into train/eval
    split = dataset.train_test_split(test_size=0.15, seed=42)
    train_data = split["train"]
    eval_data = split["test"]
    
    print(f"  Training examples: {len(train_data)}")
    print(f"  Evaluation examples: {len(eval_data)}")

    # Tokenize
    print("\nTokenizing datasets...")
    train_tokenized = manager.tokenize_dataset(train_data, max_length=512)
    eval_tokenized = manager.tokenize_dataset(eval_data, max_length=512)

    # Train
    print("\nStarting training...")
    try:
        manager.train(
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            num_epochs=3,
            batch_size=2,  # Small batch for memory efficiency
            learning_rate=2e-4,
            save_steps=50,
            logging_steps=10,
            gradient_accumulation_steps=8,
            output_name="python_assistant",
        )

        # Save the adapter
        print("\nSaving LoRA adapter...")
        manager.save_adapter("python_assistant_adapter")
        
        print("\n✓ Fine-tuning complete!")
        print("  Adapter saved to: ./fine_tuned_models/python_assistant_adapter")

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()


def step3_test_finetuned_model():
    """
    Step 3: Test the fine-tuned model.
    """
    if not FINETUNING_AVAILABLE:
        print("\n⚠ Skipping testing - dependencies not installed")
        return

    print("\n" + "="*70)
    print("STEP 3: Testing Fine-Tuned Model")
    print("="*70 + "\n")

    adapter_path = "./fine_tuned_models/python_assistant_adapter"
    
    if not os.path.exists(adapter_path):
        print(f"⚠ Adapter not found: {adapter_path}")
        print("  Run step2_finetune_model() first")
        return

    # Load the fine-tuned assistant
    print(f"Loading adapter from {adapter_path}...")
    assistant = create_finetuned_assistant(
        adapter_path=adapter_path,
        base_model="microsoft/phi-2",
        temperature=0.7,
        max_new_tokens=200,
    )

    # Test prompts
    test_prompts = [
        "User: What are Python decorators?\n\nAssistant:",
        "User: Explain async/await in Python\n\nAssistant:",
        "User: How do I use type hints?\n\nAssistant:",
        "User: What is a list comprehension?\n\nAssistant:",
    ]

    print("\nTesting fine-tuned model:\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"{'─'*70}")
        print(f"\n{prompt}")
        
        try:
            response = assistant.generate_text(prompt)
            # Extract only the assistant's response (after the prompt)
            if prompt in response:
                response = response.split(prompt)[1].strip()
            print(f"\n{response}\n")
        except Exception as e:
            print(f"\n✗ Error: {e}\n")

    print("✓ Testing complete!")


def step4_integrate_into_workflow():
    """
    Step 4: Show how to integrate fine-tuned model into your existing workflow.
    """
    print("\n" + "="*70)
    print("STEP 4: Integration Ideas")
    print("="*70 + "\n")

    integration_examples = """
    Here are ways to integrate your fine-tuned model:

    1. HYBRID APPROACH - Use both Ollama and fine-tuned model:
       
       from ollama_wrapper import OllamaWrapper, create_finetuned_assistant
       
       # Use Ollama for general chat
       ollama = OllamaWrapper()
       
       # Use fine-tuned model for specialized queries
       specialist = create_finetuned_assistant("./fine_tuned_models/my_adapter")
       
       def smart_router(query):
           if "python" in query.lower() or "code" in query.lower():
               return specialist.generate_text(f"User: {query}\\n\\nAssistant:")
           else:
               return ollama.chat(query)

    2. EXPORT TO OLLAMA - Convert to GGUF and use with Ollama:
       
       # (Future feature - coming soon)
       manager.export_to_gguf("./my_model.gguf")
       # Then: ollama create my-model -f Modelfile
       
    3. API WRAPPER - Create a FastAPI endpoint:
       
       from fastapi import FastAPI
       from ollama_wrapper import create_finetuned_assistant
       
       app = FastAPI()
       assistant = create_finetuned_assistant("./my_adapter")
       
       @app.post("/chat")
       def chat(message: str):
           return assistant.generate_text(f"User: {message}\\n\\nAssistant:")

    4. CONTINUAL LEARNING - Regularly update your model:
       
       # Collect new conversations
       wrapper = OllamaWrapper(session_id="feedback_2024_01")
       # ... interact with users ...
       
       # Fine-tune on new + old data
       manager = FineTuningManager()
       manager.load_model()
       manager.setup_lora()
       dataset = manager.load_training_data_from_memory()
       # ... train ...

    5. SPECIALIZED ASSISTANTS - Create multiple adapters:
       
       assistants = {
           "python": create_finetuned_assistant("./adapters/python"),
           "javascript": create_finetuned_assistant("./adapters/javascript"),
           "devops": create_finetuned_assistant("./adapters/devops"),
       }
       
       def get_assistant(topic):
           return assistants.get(topic, default_assistant)
    """

    print(integration_examples)


def run_full_pipeline():
    """
    Run the complete pipeline: create data -> fine-tune -> test.
    """
    print("\n" + "="*70)
    print("COMPLETE FINE-TUNING PIPELINE")
    print("="*70)

    try:
        step1_create_training_data()
        step2_finetune_model()
        step3_test_finetuned_model()
        step4_integrate_into_workflow()
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE!")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def interactive_menu():
    """
    Interactive menu for choosing what to run.
    """
    menu = """
╔════════════════════════════════════════════════════════════════════╗
║         OllamaWrapper + Fine-Tuning Integration Example           ║
╚════════════════════════════════════════════════════════════════════╝

Choose an option:

  1. Create training data (Step 1)
  2. Fine-tune model (Step 2)
  3. Test fine-tuned model (Step 3)
  4. Show integration ideas (Step 4)
  5. Run full pipeline (Steps 1-4)
  
  0. Exit

Your choice: """

    while True:
        try:
            choice = input(menu).strip()

            if choice == "0":
                print("\nGoodbye! 👋")
                break
            elif choice == "1":
                step1_create_training_data()
            elif choice == "2":
                step2_finetune_model()
            elif choice == "3":
                step3_test_finetuned_model()
            elif choice == "4":
                step4_integrate_into_workflow()
            elif choice == "5":
                run_full_pipeline()
            else:
                print("\n⚠ Invalid choice. Please select 0-5.")

            input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command-line mode
        command = sys.argv[1]
        if command == "step1":
            step1_create_training_data()
        elif command == "step2":
            step2_finetune_model()
        elif command == "step3":
            step3_test_finetuned_model()
        elif command == "step4":
            step4_integrate_into_workflow()
        elif command == "full":
            run_full_pipeline()
        else:
            print("Usage: python example_finetuning_integration.py [step1|step2|step3|step4|full]")
    else:
        # Interactive mode
        interactive_menu()
