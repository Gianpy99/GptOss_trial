"""
🔍 System Check - Verifica setup Ollama Wrapper
"""

import sys
import subprocess
from pathlib import Path
import importlib

def check_symbol(success: bool):
    return "✅" if success else "❌"

def check_python_version():
    """Verifica versione Python."""
    version = sys.version_info
    required = (3, 8)
    success = version >= required
    
    print(f"{check_symbol(success)} Python {version.major}.{version.minor}.{version.micro}", end="")
    if not success:
        print(f" (required: >={required[0]}.{required[1]})")
    else:
        print()
    
    return success

def check_package(package_name: str, import_name: str = None):
    """Verifica se un package è installato."""
    if import_name is None:
        import_name = package_name
    
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {package_name} ({version})")
        return True
    except ImportError:
        print(f"❌ {package_name} (not installed)")
        return False

def check_ollama():
    """Verifica se Ollama è installato e running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            models = result.stdout.strip().split('\n')[1:]
            print(f"✅ Ollama running ({len(models)} models)")
            return True
        else:
            print("❌ Ollama not running")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama not installed")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  Ollama timeout")
        return False

def check_cuda():
    """Verifica supporto CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"✅ CUDA {cuda_version} - {device_name}")
            return True
        else:
            print("⚠️  CUDA not available (CPU only)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_files():
    """Verifica presenza file essenziali."""
    files = {
        "ollama_cli.py": "CLI principale",
        "ui_multi_model.py": "UI multi-modello",
        "quick_train.py": "Training script",
        "combine_datasets.py": "Dataset combiner",
        "wrapper.py": "Core wrapper",
        "CLI_GUIDE.md": "Documentazione",
        "DEMO_COMPLETE.md": "Demo completa"
    }
    
    all_ok = True
    for file, desc in files.items():
        exists = Path(file).exists()
        print(f"{check_symbol(exists)} {file} - {desc}")
        all_ok = all_ok and exists
    
    return all_ok

def check_directories():
    """Verifica directory necessarie."""
    dirs = {
        ".venv_training": "Virtual environment",
        "finetuning_projects": "Training projects",
        "fine_tuned_models": "Merged models",
        "ollama_sessions": "Session storage"
    }
    
    all_ok = True
    for dir_name, desc in dirs.items():
        exists = Path(dir_name).exists()
        print(f"{check_symbol(exists)} {dir_name}/ - {desc}")
        all_ok = all_ok and exists
    
    return all_ok

def check_datasets():
    """Verifica dataset disponibili."""
    datasets = list(Path('.').glob('*data*.json'))
    datasets += list(Path('.').glob('*train*.json'))
    
    print(f"\n📊 Datasets found: {len(datasets)}")
    for ds in datasets:
        print(f"  • {ds.name}")
    
    return len(datasets) > 0

def main():
    print("="*70)
    print("  🔍 OLLAMA WRAPPER - SYSTEM CHECK")
    print("="*70)
    
    print("\n📦 Python & Packages:")
    checks = []
    checks.append(check_python_version())
    
    # Essential packages
    essential = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("gradio", "gradio"),
        ("requests", "requests")
    ]
    
    for pkg, import_name in essential:
        checks.append(check_package(pkg, import_name))
    
    print("\n🖥️  System:")
    checks.append(check_ollama())
    checks.append(check_cuda())
    
    print("\n📁 Files:")
    checks.append(check_files())
    
    print("\n📂 Directories:")
    checks.append(check_directories())
    
    print("\n📊 Datasets:")
    has_datasets = check_datasets()
    
    print("\n" + "="*70)
    
    total = len(checks)
    passed = sum(checks)
    
    if passed == total:
        print("✅ ALL CHECKS PASSED!")
        print("\n🚀 Ready to use! Try:")
        print("   .\\ollama-cli.ps1 list")
        print("   .\\ollama-cli.ps1 ui")
    elif passed >= total * 0.7:
        print(f"⚠️  PARTIAL: {passed}/{total} checks passed")
        print("\n💡 You can still use the system, but some features may not work.")
        print("   Install missing packages with: pip install -r requirements-finetuning.txt")
    else:
        print(f"❌ FAILED: Only {passed}/{total} checks passed")
        print("\n🔧 Setup required:")
        print("   1. Install Python 3.8+")
        print("   2. pip install -r requirements-finetuning.txt")
        print("   3. Install Ollama: https://ollama.ai")
    
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
