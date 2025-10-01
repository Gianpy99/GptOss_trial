"""Test veloce della funzione create_f1_training_data"""
import sys
sys.path.insert(0, '.')

from demo_f1_finetuning import download_f1_dataset, create_f1_training_data

print("📥 Downloading F1 dataset...")
rows = download_f1_dataset()

if rows:
    print(f"✓ Got {len(rows)} rows\n")
    
    print("🔧 Creating training data...")
    training = create_f1_training_data(rows, "test_training.json")
    
    print(f"\n✓ SUCCESS! Created {len(training)} training examples")
    print(f"✓ File saved: test_training.json")
    
    # Show first 3 examples
    print("\nFirst 3 examples:")
    for i, ex in enumerate(training[:3], 1):
        print(f"\n{i}. Q: {ex['instruction']}")
        print(f"   A: {ex['output']}")
else:
    print("❌ Failed to download dataset")
