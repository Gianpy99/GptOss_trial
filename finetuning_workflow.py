"""
Fine-Tuning Workflow - Modular and Reusable
============================================

This workflow is designed to work across two computers:
1. TRAINING COMPUTER: Fine-tune model and export adapter
2. INFERENCE COMPUTER: Import adapter and use with Ollama

Steps:
  [Training Computer]
  1. Download dataset (any Hugging Face dataset or custom JSON)
  2. Fine-tune model with LoRA
  3. Export adapter (small file ~20-50MB)
  
  [Transfer adapter file to inference computer]
  
  [Inference Computer]
  4. Convert adapter to GGUF format
  5. Import into Ollama
  6. Use with OllamaWrapper

This is a generic workflow - works with:
  - Formula 1 dataset (example)
  - Your custom datasets
  - Any Hugging Face dataset

Usage:
  # Training:
  python finetuning_workflow.py train --dataset f1 --model gemma3:4b
  
  # Inference:
  python finetuning_workflow.py deploy --adapter ./adapters/f1_adapter
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path

# Force UTF-8 encoding for Windows terminals
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load Hugging Face token from .env file (if exists)
def load_hf_token():
    """Load HF token from .env file or environment variable."""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('HF_TOKEN='):
                    token = line.split('=', 1)[1].strip()
                    if token and not token.startswith('hf_xxx'):
                        os.environ['HF_TOKEN'] = token
                        os.environ['HUGGING_FACE_HUB_TOKEN'] = token
                        print("✓ Loaded HF token from .env file")
                        return token
    
    # Fallback to environment variable
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        print("✓ Using HF token from environment variable")
        return token
    
    print("⚠️  No HF token found. Gated models (Gemma, Llama) will fail.")
    print("   Create .env file with: HF_TOKEN=hf_xxxxx")
    return None

# Load token at startup
load_hf_token()

# Import OllamaWrapper directly to avoid loading finetuning module at startup
from src.ollama_wrapper.wrapper import OllamaWrapper


class FineTuningWorkflow:
    """
    Manages the complete fine-tuning workflow across training and inference.
    """
    
    def __init__(self, model_name="gemma3:4b", base_dir="./finetuning_projects"):
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Extract HF model name from Ollama model name
        # Note: Gemma and Llama require HuggingFace authentication
        # You must:
        #   1. Accept license at https://huggingface.co/google/gemma-3-4b-it
        #   2. Login: huggingface-cli login
        #   3. Or set HF_TOKEN in .env file
        self.hf_model_map = {
            # Gemma models (GATED - requires HF authentication)
            "gemma3:4b": "google/gemma-3-4b-it",  # Gemma 3 4B Instruction Tuned
            "gemma3:8b": "google/gemma-2-9b",     # Gemma 2 9B
            "gemma2:2b": "google/gemma-2-2b",     # Gemma 2 2B
            
            # Llama models (GATED - requires HF authentication)
            "llama3:8b": "meta-llama/Llama-3.1-8B",
            "llama3.1:8b": "meta-llama/Llama-3.1-8B",
            
            # Phi models (OPEN - no auth needed)
            "phi3:4b": "microsoft/phi-2",
            "phi2": "microsoft/phi-2",
            "phi": "microsoft/phi-2",
        }
    
    def get_hf_model_name(self):
        """Get the Hugging Face model name for the Ollama model."""
        # Try exact match
        if self.model_name in self.hf_model_map:
            return self.hf_model_map[self.model_name]
        
        # Try base name match (e.g., "gemma3:4b-instruct" -> "gemma3")
        base_name = self.model_name.split(":")[0]
        for key in self.hf_model_map:
            if key.startswith(base_name):
                return self.hf_model_map[key]
        
        # Default fallback
        print(f"⚠️  Warning: Unknown model '{self.model_name}', using microsoft/phi-2")
        return "microsoft/phi-2"
    
    def download_dataset(self, dataset_name, limit=100):
        """
        Download dataset from Hugging Face.
        
        Supports:
        - Hugging Face dataset names (e.g., "Vadera007/Formula_1_Dataset")
        - Local JSON files
        - Custom format
        """
        print(f"\n📥 Downloading dataset: {dataset_name}")
        
        if os.path.exists(dataset_name):
            # Local file
            print(f"✓ Using local file: {dataset_name}")
            with open(dataset_name, "r", encoding="utf-8") as f:
                return json.load(f)
        
        # Try Hugging Face
        url = "https://datasets-server.huggingface.co/first-rows"
        params = {
            "dataset": dataset_name,
            "config": "default",
            "split": "train"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "rows" in data:
                rows = [row["row"] for row in data["rows"][:limit]]
                print(f"✓ Downloaded {len(rows)} rows from Hugging Face")
                return rows
            else:
                print(f"❌ Unexpected response format from Hugging Face")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return None
    
    def create_training_data(self, rows, dataset_type="generic"):
        """
        Convert raw dataset rows to instruction-following format.
        
        Args:
            rows: List of data rows
            dataset_type: Type of dataset ("f1", "generic", "qa", etc.)
        
        Returns:
            List of training examples in format:
            {"instruction": "...", "output": "..."}
        """
        print(f"\n🔄 Converting {len(rows)} rows to training format...")
        
        if dataset_type == "f1":
            return self._create_f1_training_data(rows)
        elif dataset_type == "qa":
            return self._create_qa_training_data(rows)
        else:
            return self._create_generic_training_data(rows)
    
    def _create_f1_training_data(self, rows):
        """Create F1-specific training data (example from demo)."""
        # Accumulate statistics per driver
        drivers_stats = {}
        
        for row in rows:
            driver = row.get("Driver", "Unknown")
            team = row.get("Team", "Unknown")
            avg_lap = row.get("AvgLapTime")
            laps = row.get("LapsCompleted")
            finish_pos = row.get("RaceFinishPosition")
            
            if driver not in drivers_stats:
                drivers_stats[driver] = {
                    "team": team,
                    "avg_lap_times": [],
                    "total_laps": 0,
                    "finish_positions": []
                }
            
            # Only add non-None values
            if avg_lap is not None:
                drivers_stats[driver]["avg_lap_times"].append(avg_lap)
            if laps is not None:
                drivers_stats[driver]["total_laps"] += laps
            if finish_pos is not None:
                drivers_stats[driver]["finish_positions"].append(finish_pos)
        
        # Create training examples
        training_data = []
        
        for driver, stats in drivers_stats.items():
            if not stats["avg_lap_times"]:  # Skip drivers without data
                continue
            
            team = stats["team"]
            avg_lap = sum(stats["avg_lap_times"]) / len(stats["avg_lap_times"])
            total_laps = stats["total_laps"]
            
            avg_finish = None
            if stats["finish_positions"]:
                avg_finish = sum(stats["finish_positions"]) / len(stats["finish_positions"])
            
            # Training examples
            training_data.append({
                "instruction": f"What team does {driver} drive for in Formula 1?",
                "output": f"{driver} drives for {team} in Formula 1."
            })
            
            training_data.append({
                "instruction": f"What is {driver}'s average lap time?",
                "output": f"{driver}'s average lap time is approximately {avg_lap:.2f} seconds."
            })
            
            if avg_finish:
                training_data.append({
                    "instruction": f"How does {driver} typically perform in races?",
                    "output": f"{driver} typically finishes around position {avg_finish:.1f} in races, driving for {team}."
                })
        
        print(f"✓ Created {len(training_data)} training examples")
        return training_data
    
    def _create_qa_training_data(self, rows):
        """Create QA-pair training data."""
        training_data = []
        for row in rows:
            if "question" in row and "answer" in row:
                training_data.append({
                    "instruction": row["question"],
                    "output": row["answer"]
                })
        return training_data
    
    def _create_generic_training_data(self, rows):
        """Create generic training data from any format."""
        print("⚠️  Using generic format - may need customization")
        training_data = []
        
        for i, row in enumerate(rows):
            # Try to extract meaningful instruction-output pairs
            if isinstance(row, dict):
                # Look for common patterns
                if "text" in row:
                    training_data.append({
                        "instruction": f"Example {i+1}",
                        "output": row["text"]
                    })
                else:
                    # Convert entire row to JSON string
                    training_data.append({
                        "instruction": f"Data point {i+1}",
                        "output": json.dumps(row)
                    })
        
        return training_data
    
    def train(self, training_data, project_name, epochs=3, batch_size=4):
        """
        Fine-tune the model with LoRA.
        
        This runs on the TRAINING computer.
        """
        print("\n" + "="*70)
        print("🔥 STARTING FINE-TUNING")
        print("="*70)
        
        print("\nLoading fine-tuning libraries (may take 30-60 seconds)...")
        try:
            from src.ollama_wrapper.finetuning import FineTuningManager
        except ImportError:
            print("❌ Fine-tuning libraries not installed!")
            print("   Run: pip install -r requirements-finetuning.txt")
            return None
        
        hf_model = self.get_hf_model_name()
        print(f"\n📦 Model: {self.model_name} → {hf_model}")
        print(f"📂 Project: {project_name}")
        print(f"📊 Training examples: {len(training_data)}")
        print(f"⚙️  Epochs: {epochs}, Batch size: {batch_size}")
        
        # Create project directory
        project_dir = self.base_dir / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Save training data
        training_file = project_dir / "training_data.json"
        with open(training_file, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved training data to {training_file}")
        
        # Initialize fine-tuning manager
        adapter_dir = project_dir / "adapter"
        
        # Disable 4-bit quantization if no CUDA GPU available
        import torch
        use_4bit = torch.cuda.is_available()
        if not use_4bit:
            print("⚠️  No CUDA GPU detected - using FP16 mode (slower but works on CPU)")
        
        manager = FineTuningManager(
            model_name=hf_model,
            output_dir=str(adapter_dir),
            use_4bit=use_4bit  # Auto-detect GPU
        )
        
        print("\n⏳ Loading model (this may take a few minutes)...")
        manager.load_model()
        print("✓ Model loaded")
        
        print("\n⏳ Setting up LoRA...")
        manager.setup_lora(r=16, lora_alpha=32, lora_dropout=0.05)
        print("✓ LoRA configured")
        
        print("\n⏳ Preparing dataset...")
        from datasets import Dataset
        dataset = Dataset.from_list(training_data)
        tokenized = manager.tokenize_dataset(dataset)
        print("✓ Dataset tokenized")
        
        print("\n🔥 Starting training...")
        print("   This will take several minutes depending on your hardware")
        print("   Watch for progress bars below:\n")
        
        try:
            manager.train(
                tokenized,
                num_epochs=epochs,
                per_device_batch_size=batch_size,
                learning_rate=2e-4
            )
            print("\n✓ Training completed successfully!")
            
            # Save adapter
            adapter_path = adapter_dir / "final_adapter"
            manager.save_adapter(str(adapter_path))
            print(f"✓ Adapter saved to: {adapter_path}")
            
            # Save metadata
            metadata = {
                "project_name": project_name,
                "base_model": hf_model,
                "ollama_model": self.model_name,
                "training_examples": len(training_data),
                "epochs": epochs,
                "adapter_path": str(adapter_path),
                "date": datetime.now().isoformat()
            }
            
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Metadata saved to: {metadata_file}")
            
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETE!")
            print("="*70)
            print(f"\n📦 Adapter location: {adapter_path}")
            print(f"📊 Size: ~20-50MB (check folder)")
            print(f"\n📤 NEXT STEPS:")
            print(f"   1. Copy the entire '{project_name}' folder to your inference computer")
            print(f"   2. Run: python finetuning_workflow.py deploy --project {project_name}")
            
            return str(adapter_path)
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def deploy(self, project_name):
        """
        Deploy the fine-tuned adapter to Ollama.
        
        This runs on the INFERENCE computer.
        """
        print("\n" + "="*70)
        print("🚀 DEPLOYING ADAPTER TO OLLAMA")
        print("="*70)
        
        project_dir = self.base_dir / project_name
        if not project_dir.exists():
            print(f"❌ Project not found: {project_dir}")
            return False
        
        metadata_file = project_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"❌ Metadata not found: {metadata_file}")
            return False
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        adapter_path = Path(metadata["adapter_path"])
        if not adapter_path.exists():
            print(f"❌ Adapter not found: {adapter_path}")
            return False
        
        print(f"\n📦 Project: {metadata['project_name']}")
        print(f"🤖 Base model: {metadata['ollama_model']}")
        print(f"📂 Adapter: {adapter_path}")
        
        print("\n⚠️  TODO: Implement GGUF conversion and Ollama import")
        print("   This requires:")
        print("   1. Convert adapter to GGUF format")
        print("   2. Create Modelfile")
        print("   3. Run: ollama create <model-name> -f Modelfile")
        
        # For now, return the adapter path for manual processing
        return str(adapter_path)
    
    def test_comparison(self, project_name, test_questions):
        """
        Test base model vs fine-tuned model.
        
        Runs on INFERENCE computer after deployment.
        """
        print("\n" + "="*70)
        print("📊 TESTING: BASE vs FINE-TUNED")
        print("="*70)
        
        metadata_file = self.base_dir / project_name / "metadata.json"
        if not metadata_file.exists():
            print(f"❌ Project metadata not found")
            return
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        base_model = metadata["ollama_model"]
        finetuned_model = f"{base_model.replace(':', '-')}-{project_name}"
        
        print(f"\n🤖 Base model: {base_model}")
        print(f"🎯 Fine-tuned: {finetuned_model}")
        
        # Test base model
        print("\n" + "-"*70)
        print("Testing BASE MODEL:")
        print("-"*70)
        
        wrapper_base = OllamaWrapper(model=base_model)
        
        base_responses = []
        for i, question in enumerate(test_questions, 1):
            print(f"\nQ{i}: {question}")
            try:
                response = wrapper_base.chat(question)
                answer = response.get("message", {}).get("content", "No response")
                print(f"A{i}: {answer}")
                base_responses.append(answer)
            except Exception as e:
                print(f"A{i}: ❌ Error: {e}")
                base_responses.append(f"Error: {e}")
        
        # Test fine-tuned model
        print("\n" + "-"*70)
        print("Testing FINE-TUNED MODEL:")
        print("-"*70)
        
        try:
            wrapper_finetuned = OllamaWrapper(model=finetuned_model)
            
            finetuned_responses = []
            for i, question in enumerate(test_questions, 1):
                print(f"\nQ{i}: {question}")
                try:
                    response = wrapper_finetuned.chat(question)
                    answer = response.get("message", {}).get("content", "No response")
                    print(f"A{i}: {answer}")
                    finetuned_responses.append(answer)
                except Exception as e:
                    print(f"A{i}: ❌ Error: {e}")
                    finetuned_responses.append(f"Error: {e}")
        
        except Exception as e:
            print(f"❌ Could not load fine-tuned model: {e}")
            print(f"   Make sure you've imported it into Ollama first")
            finetuned_responses = [f"Model not found: {e}"] * len(test_questions)
        
        # Comparison
        print("\n" + "="*70)
        print("📊 COMPARISON")
        print("="*70)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Question {i} ---")
            print(f"Q: {question}")
            print(f"\n[BASE]: {base_responses[i-1][:200]}...")
            print(f"\n[TUNED]: {finetuned_responses[i-1][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning Workflow Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new adapter")
    train_parser.add_argument("--dataset", required=True, help="Dataset name or path")
    train_parser.add_argument("--project", required=True, help="Project name")
    train_parser.add_argument("--model", default="gemma3:4b", help="Ollama model name")
    train_parser.add_argument("--type", default="generic", help="Dataset type (f1, qa, generic)")
    train_parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--limit", type=int, default=100, help="Dataset row limit")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy adapter to Ollama")
    deploy_parser.add_argument("--project", required=True, help="Project name")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test base vs fine-tuned")
    test_parser.add_argument("--project", required=True, help="Project name")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "train":
        workflow = FineTuningWorkflow(model_name=args.model)
        
        # Download dataset
        rows = workflow.download_dataset(args.dataset, limit=args.limit)
        if not rows:
            print("❌ Failed to download dataset")
            return
        
        # Create training data
        training_data = workflow.create_training_data(rows, dataset_type=args.type)
        if not training_data:
            print("❌ Failed to create training data")
            return
        
        # Train
        adapter_path = workflow.train(
            training_data,
            project_name=args.project,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        if adapter_path:
            print("\n✅ Success! Adapter ready for deployment.")
    
    elif args.command == "deploy":
        workflow = FineTuningWorkflow()
        workflow.deploy(args.project)
    
    elif args.command == "test":
        # Example test questions for F1
        test_questions = [
            "What team does VER drive for in Formula 1?",
            "Which drivers race for Red Bull Racing?",
            "What is VER's average lap time?",
            "Who is the fastest driver based on lap times?",
        ]
        
        workflow = FineTuningWorkflow()
        workflow.test_comparison(args.project, test_questions)


if __name__ == "__main__":
    from datetime import datetime
    main()
