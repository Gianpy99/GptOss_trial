"""
Quick Start Example - Fine-tuning in 5 minutes!

This is the simplest possible example of fine-tuning with OllamaWrapper.
Perfect for getting started quickly.
"""

from ollama_wrapper import OllamaWrapper

# Check if fine-tuning is available
try:
    from ollama_wrapper import FineTuningManager
    print("✓ Fine-tuning dependencies installed")
except ImportError:
    print("✗ Fine-tuning not available. Install with:")
    print("  pip install -r requirements-finetuning.txt")
    exit(1)


def main():
    print("\n" + "="*60)
    print("Quick Start: Fine-Tuning in 5 Minutes")
    print("="*60 + "\n")

    # Step 1: Create some training conversations
    print("Step 1: Creating training conversations...\n")
    
    wrapper = OllamaWrapper(session_id="quick_training")
    
    questions = [
        "What is Python?",
        "How do I create a list in Python?",
        "What are Python functions?",
        "Explain Python loops",
        "What is a dictionary in Python?",
    ]
    
    for q in questions:
        print(f"  Q: {q}")
        try:
            answer = wrapper.chat(q)
            print(f"  A: {answer[:60]}...\n")
        except Exception as e:
            print(f"  Error: {e}\n")

    # Step 2: Fine-tune
    print("\nStep 2: Fine-tuning model...\n")
    
    manager = FineTuningManager(
        model_name="microsoft/phi-2",  # Small, fast model
        use_4bit=True,                 # Memory efficient
    )
    
    print("Loading model...")
    manager.load_model()
    
    print("Setting up LoRA...")
    manager.setup_lora(r=8, lora_alpha=16)  # Small, fast config
    
    print("Loading training data...")
    dataset = manager.load_training_data_from_memory(
        session_ids=["quick_training"]
    )
    
    if len(dataset) < 3:
        print(f"⚠ Only {len(dataset)} examples. Need at least 3.")
        print("Run this script again to add more conversations.")
        return
    
    print(f"Found {len(dataset)} training examples")
    
    print("\nTokenizing...")
    tokenized = manager.tokenize_dataset(dataset, max_length=256)
    
    print("\nTraining (this will take a few minutes)...")
    manager.train(
        train_dataset=tokenized,
        num_epochs=2,          # Quick training
        batch_size=1,          # Small batch
        learning_rate=2e-4,
        save_steps=10,
        logging_steps=5,
        output_name="quick_model",
    )
    
    print("\nSaving adapter...")
    manager.save_adapter("quick_adapter")
    
    # Step 3: Test it
    print("\n" + "="*60)
    print("Step 3: Testing the fine-tuned model")
    print("="*60 + "\n")
    
    test_prompt = "User: What is Python?\n\nAssistant:"
    print(f"Prompt: {test_prompt}\n")
    
    response = manager.generate_text(
        test_prompt,
        max_new_tokens=100,
        temperature=0.7,
    )
    
    print(f"Response:\n{response}\n")
    
    print("="*60)
    print("✓ Done! Your model is saved at:")
    print("  ./fine_tuned_models/quick_adapter")
    print("\nTo use it again:")
    print("  from ollama_wrapper import create_finetuned_assistant")
    print("  assistant = create_finetuned_assistant('./fine_tuned_models/quick_adapter')")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
