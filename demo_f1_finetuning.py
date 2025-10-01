"""
Formula 1 Fine-Tuning Demo
==========================

Dimostra il miglioramento del modello dopo fine-tuning su dati Formula 1.

Steps:
1. Scarica dataset F1 da Hugging Face
2. Testa modello base su domande F1 (risposta generica/errata)
3. Fine-tune il modello con dati F1
4. Ritesta le stesse domande (risposta accurata!)

Requirements:
    pip install -r requirements-finetuning.txt
"""

import sys
import os
import json
import requests

# Force UTF-8 encoding for Windows terminals
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ollama_wrapper import OllamaWrapper

try:
    from src.ollama_wrapper import FineTuningManager, create_finetuned_assistant
    FINETUNING_AVAILABLE = True
except ImportError:
    FINETUNING_AVAILABLE = False
    print("❌ Fine-tuning not available. Install with:")
    print("   pip install -r requirements-finetuning.txt")
    sys.exit(1)


def download_f1_dataset(limit=100):
    """
    Scarica il dataset Formula 1 da Hugging Face.
    
    Returns:
        list: Lista di righe del dataset
    """
    print("📥 Downloading Formula 1 dataset from Hugging Face...")
    
    url = "https://datasets-server.huggingface.co/first-rows"
    params = {
        "dataset": "Vadera007/Formula_1_Dataset",
        "config": "default",
        "split": "train"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        rows = data.get("rows", [])
        print(f"✓ Downloaded {len(rows)} rows from F1 dataset")
        
        return rows
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return []


def create_f1_training_data(rows, output_file="f1_training_data.json"):
    """
    Converte i dati F1 in formato training.
    
    Crea coppie instruction-output basate sui dati reali.
    """
    print(f"\n🔧 Creating training data from {len(rows)} rows...")
    
    training_data = []
    
    # Analizza i dati per creare domande/risposte
    drivers_stats = {}
    teams_stats = {}
    
    for item in rows:
        row = item.get("row", {})
        driver = row.get("Driver")
        team = row.get("Team")
        avg_lap = row.get("AvgLapTime")
        laps = row.get("LapsCompleted")
        air_temp = row.get("AirTemp")
        track_temp = row.get("TrackTemp")
        rainfall = row.get("Rainfall")
        quali_pos = row.get("QualiPosition")
        finish_pos = row.get("RaceFinishPosition")
        
        # Accumula statistiche (skip None values)
        if driver and driver not in drivers_stats:
            drivers_stats[driver] = {
                "team": team,
                "avg_lap_times": [],
                "total_laps": 0,
                "positions": []
            }
        
        if driver:
            # Solo aggiungi valori non-None
            if avg_lap is not None:
                drivers_stats[driver]["avg_lap_times"].append(avg_lap)
            if laps is not None:
                drivers_stats[driver]["total_laps"] += laps
            if finish_pos is not None:
                drivers_stats[driver]["positions"].append(finish_pos)
        
        if team:
            if team not in teams_stats:
                teams_stats[team] = {"drivers": set(), "avg_laps": []}
            if driver:
                teams_stats[team]["drivers"].add(driver)
            if avg_lap is not None:
                teams_stats[team]["avg_laps"].append(avg_lap)
    
    # Crea training examples
    # Tipo 1: Info su piloti
    for driver, stats in drivers_stats.items():
        # Solo se abbiamo dati validi
        if not stats["avg_lap_times"]:
            continue
            
        avg_lap = sum(stats["avg_lap_times"]) / len(stats["avg_lap_times"])
        
        if stats["team"]:
            training_data.append({
                "instruction": f"What team does {driver} drive for in Formula 1?",
                "output": f"{driver} drives for {stats['team']} in Formula 1."
            })
        
        training_data.append({
            "instruction": f"What is {driver}'s average lap time?",
            "output": f"{driver}'s average lap time is approximately {avg_lap:.2f} seconds."
        })
        
        if stats["positions"]:
            avg_pos = sum(stats["positions"]) / len(stats["positions"])
            training_data.append({
                "instruction": f"How does {driver} typically perform in races?",
                "output": f"{driver} typically finishes around position {avg_pos:.1f} in races, driving for {stats['team']}."
            })
    
    # Tipo 2: Info su team
    for team, stats in teams_stats.items():
        if not stats["avg_laps"]:
            continue
            
        drivers_list = ", ".join(sorted(stats["drivers"]))
        avg_lap = sum(stats["avg_laps"]) / len(stats["avg_laps"])
        
        training_data.append({
            "instruction": f"Which drivers race for {team}?",
            "output": f"The drivers racing for {team} are: {drivers_list}."
        })
        
        training_data.append({
            "instruction": f"What is {team}'s average lap time performance?",
            "output": f"{team} has an average lap time of approximately {avg_lap:.2f} seconds across their drivers."
        })
    
    # Tipo 3: Domande comparative
    sorted_drivers = sorted(
        [(d, s) for d, s in drivers_stats.items() if s["avg_lap_times"]],
        key=lambda x: sum(x[1]["avg_lap_times"]) / len(x[1]["avg_lap_times"])
    )
    
    if len(sorted_drivers) >= 2:
        fastest = sorted_drivers[0]
        slowest = sorted_drivers[-1]
        fastest_avg = sum(fastest[1]["avg_lap_times"]) / len(fastest[1]["avg_lap_times"])
        
        training_data.append({
            "instruction": "Who is the fastest driver based on lap times?",
            "output": f"Based on lap times, {fastest[0]} from {fastest[1]['team']} is the fastest driver with an average of {fastest_avg:.2f} seconds."
        })
    
    # Salva in JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created {len(training_data)} training examples")
    print(f"✓ Saved to {output_file}")
    
    return training_data


def test_model_before_finetuning(model_name="microsoft/phi-2"):
    """
    Testa il modello base su domande F1 (dovrebbe dare risposte generiche).
    """
    print("\n" + "="*70)
    print("📊 STEP 1: Testing BASE MODEL (Before Fine-Tuning)")
    print("="*70 + "\n")
    
    test_questions = [
        "What team does VER drive for in Formula 1?",
        "Which drivers race for Red Bull Racing?",
        "What is VER's average lap time?",
        "Who is the fastest driver based on lap times?",
    ]
    
    print("Loading base model (this may take a few minutes)...")
    manager = FineTuningManager(model_name=model_name, use_4bit=True)
    manager.load_model()
    
    base_responses = {}
    
    print("\nAsking questions to BASE model:\n")
    for i, question in enumerate(test_questions, 1):
        print(f"Q{i}: {question}")
        prompt = f"User: {question}\n\nAssistant:"
        
        try:
            response = manager.generate_text(
                prompt,
                max_new_tokens=100,
                temperature=0.7,
            )
            # Extract only assistant's response
            if "Assistant:" in response:
                answer = response.split("Assistant:")[-1].strip()
            else:
                answer = response.split(prompt)[-1].strip()
            
            print(f"A{i}: {answer}\n")
            base_responses[question] = answer
        except Exception as e:
            print(f"A{i}: ❌ Error: {e}\n")
            base_responses[question] = f"Error: {e}"
    
    return base_responses, manager


def finetune_on_f1_data(manager, training_file="f1_training_data.json"):
    """
    Fine-tune il modello sui dati F1.
    """
    print("\n" + "="*70)
    print("🔧 STEP 2: FINE-TUNING on Formula 1 Data")
    print("="*70 + "\n")
    
    print("Setting up LoRA...")
    manager.setup_lora(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    
    print(f"\nLoading training data from {training_file}...")
    dataset = manager.load_training_data_from_json(training_file)
    
    if len(dataset) < 5:
        print(f"⚠️ Warning: Only {len(dataset)} training examples. Results may vary.")
    
    print("\nTokenizing dataset...")
    tokenized = manager.tokenize_dataset(dataset, max_length=256)
    
    print("\n🚀 Starting training...")
    print("⏱️ This will take several minutes depending on your GPU...")
    print("-" * 70)
    
    try:
        manager.train(
            train_dataset=tokenized,
            num_epochs=3,
            batch_size=2,
            learning_rate=2e-4,
            save_steps=50,
            logging_steps=10,
            gradient_accumulation_steps=4,
            output_name="f1_expert",
        )
        
        print("\n✓ Training completed successfully!")
        
        print("\nSaving adapter...")
        manager.save_adapter("f1_expert_adapter")
        print("✓ Adapter saved to: ./fine_tuned_models/f1_expert_adapter")
        
        return True
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_after_finetuning(test_questions):
    """
    Testa il modello fine-tuned sulle stesse domande.
    """
    print("\n" + "="*70)
    print("🎯 STEP 3: Testing FINE-TUNED MODEL (After Training)")
    print("="*70 + "\n")
    
    adapter_path = "./fine_tuned_models/f1_expert_adapter"
    
    if not os.path.exists(adapter_path):
        print(f"❌ Adapter not found at {adapter_path}")
        return {}
    
    print("Loading fine-tuned model...")
    assistant = create_finetuned_assistant(
        adapter_path,
        base_model="microsoft/phi-2",
        temperature=0.7,
        max_new_tokens=100,
    )
    
    finetuned_responses = {}
    
    print("\nAsking the SAME questions to FINE-TUNED model:\n")
    for i, question in enumerate(test_questions, 1):
        print(f"Q{i}: {question}")
        prompt = f"User: {question}\n\nAssistant:"
        
        try:
            response = assistant.generate_text(prompt)
            # Extract only assistant's response
            if "Assistant:" in response:
                answer = response.split("Assistant:")[-1].strip()
            else:
                answer = response.split(prompt)[-1].strip()
            
            print(f"A{i}: {answer}\n")
            finetuned_responses[question] = answer
        except Exception as e:
            print(f"A{i}: ❌ Error: {e}\n")
            finetuned_responses[question] = f"Error: {e}"
    
    return finetuned_responses


def compare_results(base_responses, finetuned_responses):
    """
    Confronta le risposte prima e dopo il fine-tuning.
    """
    print("\n" + "="*70)
    print("📊 STEP 4: COMPARISON - Before vs After Fine-Tuning")
    print("="*70 + "\n")
    
    for i, question in enumerate(base_responses.keys(), 1):
        print(f"\n{'─'*70}")
        print(f"Question {i}: {question}")
        print(f"{'─'*70}")
        
        print(f"\n🔵 BEFORE Fine-Tuning:")
        print(f"   {base_responses[question][:200]}...")
        
        if question in finetuned_responses:
            print(f"\n🟢 AFTER Fine-Tuning:")
            print(f"   {finetuned_responses[question][:200]}...")
        else:
            print(f"\n⚠️ No fine-tuned response available")
        
        print()
    
    print("="*70)
    print("✅ DEMO COMPLETED!")
    print("="*70)
    print("\n📈 Key Observations:")
    print("   • Base model gives generic/incorrect answers")
    print("   • Fine-tuned model provides specific F1 knowledge")
    print("   • The adapter is only a few MB in size")
    print("   • Training took just a few minutes")
    print("\n💡 This demonstrates how fine-tuning specializes models!")


def main():
    """
    Main workflow completo.
    """
    print("\n" + "🏎️ "*25)
    print("      FORMULA 1 FINE-TUNING DEMO")
    print("🏎️ "*25 + "\n")
    
    print("This demo will:")
    print("  1. Download F1 dataset from Hugging Face")
    print("  2. Test base model (generic answers)")
    print("  3. Fine-tune on F1 data")
    print("  4. Test fine-tuned model (specific answers)")
    print("  5. Compare results\n")
    
    input("Press Enter to start...")
    
    try:
        # Step 0: Download dataset
        rows = download_f1_dataset()
        if not rows:
            print("❌ Failed to download dataset. Exiting.")
            return
        
        # Create training data
        training_data = create_f1_training_data(rows)
        
        # Step 1: Test base model
        test_questions = [
            "What team does VER drive for in Formula 1?",
            "Which drivers race for Red Bull Racing?",
            "What is VER's average lap time?",
            "Who is the fastest driver based on lap times?",
        ]
        
        base_responses, manager = test_model_before_finetuning()
        
        print("\n⏸️ Review the base model responses above.")
        input("Press Enter to continue with fine-tuning...")
        
        # Step 2: Fine-tune
        success = finetune_on_f1_data(manager)
        
        if not success:
            print("❌ Fine-tuning failed. Cannot continue.")
            return
        
        print("\n⏸️ Fine-tuning completed!")
        input("Press Enter to test the fine-tuned model...")
        
        # Step 3: Test fine-tuned model
        finetuned_responses = test_model_after_finetuning(test_questions)
        
        # Step 4: Compare
        compare_results(base_responses, finetuned_responses)
        
        print("\n✅ Demo completed successfully!")
        print("\n💾 Files created:")
        print("   • f1_training_data.json - Training data")
        print("   • ./fine_tuned_models/f1_expert_adapter - Fine-tuned adapter")
        print("\n🚀 You can now use the F1 expert model:")
        print("   from ollama_wrapper import create_finetuned_assistant")
        print("   assistant = create_finetuned_assistant('./fine_tuned_models/f1_expert_adapter')")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
