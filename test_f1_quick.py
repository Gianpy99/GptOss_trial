"""
Quick F1 Fine-Tuning Test
=========================

Versione rapida per testare il fine-tuning con dati F1.
Ideale per demo veloci.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ollama_wrapper import FineTuningManager, create_finetuned_assistant
import requests
import json


# Test questions
TEST_QUESTIONS = [
    "What team does VER drive for?",
    "Who drives for Red Bull Racing?",
    "What is Verstappen's average lap time?",
]


def quick_test():
    print("\n🏎️ F1 Fine-Tuning Quick Test\n")
    
    # Download F1 data
    print("📥 Downloading F1 dataset...")
    url = "https://datasets-server.huggingface.co/first-rows?dataset=Vadera007%2FFormula_1_Dataset&config=default&split=train"
    response = requests.get(url, timeout=30)
    data = response.json()
    rows = data.get("rows", [])
    print(f"✓ Got {len(rows)} rows\n")
    
    # Create simple training data
    training = []
    for item in rows[:20]:  # Solo primi 20 per velocità
        row = item["row"]
        driver = row.get("Driver")
        team = row.get("Team")
        avg_lap = row.get("AvgLapTime")
        
        if driver and team:
            training.append({
                "instruction": f"What team does {driver} drive for?",
                "output": f"{driver} drives for {team}."
            })
            
            if avg_lap is not None:  # Check for None
                training.append({
                    "instruction": f"What is {driver}'s average lap time?",
                    "output": f"{driver}'s average lap time is {avg_lap:.2f} seconds."
                })
    
    # Save training data
    with open("f1_quick_train.json", "w") as f:
        json.dump(training, f, indent=2)
    print(f"✓ Created {len(training)} training examples\n")
    
    # Test base model
    print("="*60)
    print("BEFORE Fine-Tuning (Base Model)")
    print("="*60 + "\n")
    
    manager = FineTuningManager(model_name="microsoft/phi-2", use_4bit=True)
    manager.load_model()
    
    print("Testing base model:\n")
    for q in TEST_QUESTIONS:
        prompt = f"User: {q}\n\nAssistant:"
        response = manager.generate_text(prompt, max_new_tokens=80, temperature=0.7)
        answer = response.split("Assistant:")[-1].strip() if "Assistant:" in response else response
        print(f"Q: {q}")
        print(f"A: {answer[:150]}...\n")
    
    # Fine-tune
    print("\n" + "="*60)
    print("Fine-Tuning...")
    print("="*60 + "\n")
    
    manager.setup_lora(r=8, lora_alpha=16)
    dataset = manager.load_training_data_from_json("f1_quick_train.json")
    tokenized = manager.tokenize_dataset(dataset, max_length=256)
    
    manager.train(
        train_dataset=tokenized,
        num_epochs=2,
        batch_size=1,
        learning_rate=2e-4,
        save_steps=20,
        logging_steps=5,
        output_name="f1_quick"
    )
    
    manager.save_adapter("f1_quick_adapter")
    print("\n✓ Training complete!\n")
    
    # Test fine-tuned
    print("="*60)
    print("AFTER Fine-Tuning (F1 Expert Model)")
    print("="*60 + "\n")
    
    assistant = create_finetuned_assistant(
        "./fine_tuned_models/f1_quick_adapter",
        base_model="microsoft/phi-2",
        temperature=0.7
    )
    
    print("Testing fine-tuned model:\n")
    for q in TEST_QUESTIONS:
        prompt = f"User: {q}\n\nAssistant:"
        response = assistant.generate_text(prompt, max_new_tokens=80)
        answer = response.split("Assistant:")[-1].strip() if "Assistant:" in response else response
        print(f"Q: {q}")
        print(f"A: {answer[:150]}...\n")
    
    print("="*60)
    print("✅ Done! Compare the answers above.")
    print("="*60)


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
