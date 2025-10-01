#!/usr/bin/env python
"""
🏎️ Ollama Wrapper CLI - Unified Interface
Gestione completa: training, deployment, testing, e web UI
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Optional, List
import argparse

# Colori per output (funziona su Windows 10+)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Stampa header colorato."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_success(text: str):
    """Stampa messaggio di successo."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    """Stampa messaggio di errore."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text: str):
    """Stampa info."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def print_warning(text: str):
    """Stampa warning."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def run_command(cmd: str, description: str, capture_output: bool = False) -> Optional[str]:
    """Esegue comando e mostra output."""
    print_info(f"Running: {description}")
    
    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print_success(description)
            return result.stdout
        else:
            print_error(f"{description} failed")
            if result.stderr:
                print(result.stderr)
            return None
    else:
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            print_success(description)
            return "OK"
        else:
            print_error(f"{description} failed")
            return None

def list_ollama_models() -> List[str]:
    """Lista modelli disponibili in Ollama."""
    result = run_command("ollama list", "Fetching Ollama models", capture_output=True)
    if result:
        lines = result.strip().split('\n')[1:]  # Skip header
        models = [line.split()[0] for line in lines if line.strip()]
        return models
    return []

def list_datasets() -> List[str]:
    """Lista dataset JSON disponibili."""
    datasets = list(Path('.').glob('*.json'))
    datasets = [d for d in datasets if 'data' in d.name.lower() or 'train' in d.name.lower()]
    return [str(d) for d in datasets]

def list_projects() -> List[str]:
    """Lista progetti di training."""
    projects_dir = Path('finetuning_projects')
    if projects_dir.exists():
        return [str(p.name) for p in projects_dir.iterdir() if p.is_dir()]
    return []

def cmd_train(args):
    """Comando: Train un nuovo modello."""
    print_header("🏋️  TRAINING")
    
    dataset = args.dataset
    project = args.project or Path(dataset).stem.replace('_data', '').replace('_training', '')
    limit = args.limit or 50
    epochs = args.epochs or 2
    
    print_info(f"Dataset: {dataset}")
    print_info(f"Project: {project}")
    print_info(f"Examples: {limit}")
    print_info(f"Epochs: {epochs}")
    
    cmd = f".venv_training\\Scripts\\activate && python quick_train.py {dataset} {project} {limit}"
    
    if not run_command(cmd, "Training model"):
        return False
    
    print_success(f"Training completed! Project: {project}")
    
    # Ask if merge
    if not args.no_merge:
        print_info("\nMerging adapter with base model...")
        return cmd_merge(argparse.Namespace(project=project, no_deploy=args.no_deploy))
    
    return True

def cmd_merge(args):
    """Comando: Merge adapter con modello base."""
    print_header("🔄 MERGE ADAPTER")
    
    project = args.project
    
    if not project:
        projects = list_projects()
        if not projects:
            print_error("No training projects found")
            return False
        
        print("Available projects:")
        for i, p in enumerate(projects, 1):
            print(f"  {i}. {p}")
        
        choice = input("\nSelect project (number): ").strip()
        try:
            project = projects[int(choice) - 1]
        except:
            print_error("Invalid choice")
            return False
    
    adapter_path = Path(f"finetuning_projects/{project}/adapter")
    if not adapter_path.exists():
        print_error(f"Adapter not found: {adapter_path}")
        return False
    
    output_dir = Path(f"fine_tuned_models/{project}_merged")
    
    # Script di merge inline
    merge_script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

print("🔄 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.float16,
    device_map="auto"
)

print("🔄 Loading adapter...")
model = PeftModel.from_pretrained(base_model, "{adapter_path}")

print("🔄 Merging...")
model = model.merge_and_unload()

print("💾 Saving merged model...")
Path("{output_dir}").mkdir(parents=True, exist_ok=True)
model.save_pretrained("{output_dir}")

print("💾 Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
tokenizer.save_pretrained("{output_dir}")

print("✅ Merge complete!")
"""
    
    temp_file = Path("_temp_merge.py")
    temp_file.write_text(merge_script, encoding='utf-8')
    
    cmd = f".venv_training\\Scripts\\activate && python _temp_merge.py"
    result = run_command(cmd, "Merging adapter with base model")
    
    temp_file.unlink(missing_ok=True)
    
    if not result:
        return False
    
    print_success(f"Merged model saved to: {output_dir}")
    
    # Ask if deploy
    if not args.no_deploy:
        print_info("\nDeploying to Ollama...")
        return cmd_deploy(argparse.Namespace(project=project, test=True))
    
    return True

def cmd_deploy(args):
    """Comando: Deploy modello su Ollama."""
    print_header("📦 DEPLOY TO OLLAMA")
    
    project = args.project
    model_name = args.name or project.replace('_', '-')
    
    merged_dir = Path(f"fine_tuned_models/{project}_merged")
    if not merged_dir.exists():
        print_error(f"Merged model not found: {merged_dir}")
        print_info("Run 'ollama-cli merge' first")
        return False
    
    # Crea Modelfile
    modelfile_content = f"""FROM {merged_dir}

TEMPLATE \"\"\"<start_of_turn>user
{{{{ .Prompt }}}}<end_of_turn>
<start_of_turn>model
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048

SYSTEM \"\"\"You are a helpful AI assistant with expertise in Formula 1 and J.R.R. Tolkien's works. Provide accurate, detailed information on these topics.\"\"\"
"""
    
    modelfile_path = Path(f"Modelfile-{project}")
    modelfile_path.write_text(modelfile_content, encoding='utf-8')
    
    print_info(f"Created Modelfile: {modelfile_path}")
    
    # Import in Ollama
    cmd = f"ollama create {model_name} -f {modelfile_path}"
    if not run_command(cmd, f"Importing model '{model_name}' to Ollama"):
        return False
    
    print_success(f"Model '{model_name}' deployed to Ollama!")
    
    # Test if requested
    if args.test:
        print_info("\nTesting model...")
        test_prompt = "Who is Lewis Hamilton?"
        cmd = f'ollama run {model_name} "{test_prompt}"'
        run_command(cmd, "Running test query")
    
    return True

def cmd_list(args):
    """Comando: Lista risorse disponibili."""
    print_header("📋 AVAILABLE RESOURCES")
    
    if args.type in ['all', 'models']:
        print(f"\n{Colors.BOLD}Ollama Models:{Colors.END}")
        models = list_ollama_models()
        if models:
            for model in models:
                print(f"  • {model}")
        else:
            print("  (none)")
    
    if args.type in ['all', 'datasets']:
        print(f"\n{Colors.BOLD}Datasets:{Colors.END}")
        datasets = list_datasets()
        if datasets:
            for ds in datasets:
                print(f"  • {ds}")
        else:
            print("  (none)")
    
    if args.type in ['all', 'projects']:
        print(f"\n{Colors.BOLD}Training Projects:{Colors.END}")
        projects = list_projects()
        if projects:
            for proj in projects:
                adapter_exists = Path(f"finetuning_projects/{proj}/adapter").exists()
                merged_exists = Path(f"fine_tuned_models/{proj}_merged").exists()
                status = []
                if adapter_exists:
                    status.append("adapter✓")
                if merged_exists:
                    status.append("merged✓")
                status_str = f" [{', '.join(status)}]" if status else ""
                print(f"  • {proj}{status_str}")
        else:
            print("  (none)")
    
    print()

def cmd_ui(args):
    """Comando: Avvia Web UI."""
    print_header("🌐 LAUNCHING WEB UI")
    
    model = args.model
    port = args.port or 7860
    
    # Usa direttamente ui_multi_model.py
    print_info(f"Starting Multi-Model UI on port {port}...")
    if model:
        print_info(f"Default model: {model}")
    
    try:
        # Avvia ui_multi_model direttamente
        cmd = f".venv_training\\Scripts\\activate && python ui_multi_model.py"
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        print_info("\nShutting down UI...")
    
    return True

def cmd_pipeline(args):
    """Comando: Pipeline completa train+merge+deploy+ui."""
    print_header("🚀 FULL PIPELINE")
    
    print_info("This will run the complete workflow:")
    print_info("  1. Train model")
    print_info("  2. Merge adapter")
    print_info("  3. Deploy to Ollama")
    print_info("  4. Launch Web UI")
    
    # Train
    train_args = argparse.Namespace(
        dataset=args.dataset,
        project=args.project,
        limit=args.limit,
        epochs=args.epochs,
        no_merge=False,
        no_deploy=False
    )
    
    if not cmd_train(train_args):
        return False
    
    # UI
    model_name = (args.project or Path(args.dataset).stem).replace('_', '-')
    ui_args = argparse.Namespace(model=model_name, port=7860)
    
    input(f"\n{Colors.GREEN}✅ Pipeline complete! Press ENTER to launch UI...{Colors.END}")
    
    cmd_ui(ui_args)
    
    return True

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='🏎️ Ollama Wrapper CLI - Unified fine-tuning and deployment tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  ollama-cli pipeline --dataset f1_data.json --project f1_expert
  
  # Train only
  ollama-cli train --dataset tolkien_data.json --limit 50
  
  # Merge existing project
  ollama-cli merge --project hybrid_expert
  
  # Deploy to Ollama
  ollama-cli deploy --project hybrid_expert --name my-model
  
  # Launch UI
  ollama-cli ui --model hybrid-expert
  
  # List resources
  ollama-cli list --type models
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--dataset', required=True, help='Dataset JSON file')
    train_parser.add_argument('--project', help='Project name (default: derived from dataset)')
    train_parser.add_argument('--limit', type=int, help='Number of examples (default: 50)')
    train_parser.add_argument('--epochs', type=int, help='Training epochs (default: 2)')
    train_parser.add_argument('--no-merge', action='store_true', help='Skip auto-merge')
    train_parser.add_argument('--no-deploy', action='store_true', help='Skip auto-deploy')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge adapter with base model')
    merge_parser.add_argument('--project', help='Project name')
    merge_parser.add_argument('--no-deploy', action='store_true', help='Skip auto-deploy')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy model to Ollama')
    deploy_parser.add_argument('--project', required=True, help='Project name')
    deploy_parser.add_argument('--name', help='Model name in Ollama (default: project name)')
    deploy_parser.add_argument('--test', action='store_true', help='Run test query after deploy')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available resources')
    list_parser.add_argument('--type', choices=['all', 'models', 'datasets', 'projects'], 
                            default='all', help='What to list')
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Launch web UI')
    ui_parser.add_argument('--model', help='Ollama model to use')
    ui_parser.add_argument('--port', type=int, default=7860, help='Port (default: 7860)')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline (train+merge+deploy+ui)')
    pipeline_parser.add_argument('--dataset', required=True, help='Dataset JSON file')
    pipeline_parser.add_argument('--project', help='Project name')
    pipeline_parser.add_argument('--limit', type=int, default=50, help='Number of examples')
    pipeline_parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Dispatch
    commands = {
        'train': cmd_train,
        'merge': cmd_merge,
        'deploy': cmd_deploy,
        'list': cmd_list,
        'ui': cmd_ui,
        'pipeline': cmd_pipeline
    }
    
    try:
        success = commands[args.command](args)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_info("\n\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
