"""
Deploy Fine-tuned Adapter to Ollama
Crea un Modelfile e registra il modello fine-tuned su Ollama
"""

import os
import sys
import io
import subprocess
import json
from pathlib import Path

# Forza UTF-8 su Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print(f"\n{'='*60}")
print(f"  🚀 DEPLOY FINE-TUNED MODEL TO OLLAMA")
print(f"{'='*60}\n")

# Configurazione
PROJECT_NAME = "f1_expert_fixed"
BASE_MODEL = "gemma3:4b"  # Modello base Ollama
DEPLOYED_MODEL_NAME = "gemma3-f1-expert:latest"  # Nome del modello fine-tuned su Ollama

ADAPTER_PATH = f"./finetuning_projects/{PROJECT_NAME}/adapter"
MODELFILE_PATH = f"./finetuning_projects/{PROJECT_NAME}/Modelfile"

print(f"⚙️ Configurazione:")
print(f"  Project: {PROJECT_NAME}")
print(f"  Base Model: {BASE_MODEL}")
print(f"  Deployed Model Name: {DEPLOYED_MODEL_NAME}")
print(f"  Adapter Path: {ADAPTER_PATH}")
print()

# Verifica adapter
if not os.path.exists(ADAPTER_PATH):
    print(f"❌ Errore: Adapter non trovato in {ADAPTER_PATH}")
    print(f"   Esegui prima il training per creare l'adapter")
    sys.exit(1)

adapter_file = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
if not os.path.exists(adapter_file):
    print(f"❌ Errore: File adapter_model.safetensors non trovato")
    sys.exit(1)

adapter_size = os.path.getsize(adapter_file) / (1024 * 1024)
print(f"✓ Adapter trovato: {adapter_file}")
print(f"  Size: {adapter_size:.1f} MB")
print()

# Verifica Ollama
print(f"🔍 Verifico Ollama...")
try:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"✓ Ollama disponibile")
    
    # Verifica se il modello base esiste
    if BASE_MODEL in result.stdout:
        print(f"✓ Modello base '{BASE_MODEL}' trovato")
    else:
        print(f"⚠️ Modello base '{BASE_MODEL}' non trovato")
        print(f"   Provo a scaricarlo...")
        subprocess.run(["ollama", "pull", BASE_MODEL], check=True)
        print(f"✓ Modello base scaricato")
except FileNotFoundError:
    print(f"❌ Errore: Ollama non trovato")
    print(f"   Installa Ollama da: https://ollama.ai")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"❌ Errore durante verifica Ollama: {e}")
    sys.exit(1)
print()

# Crea Modelfile
print(f"📝 Creo Modelfile...")
# Ollama richiede la directory dell'adapter, non il file specifico
adapter_dir_abs = os.path.abspath(ADAPTER_PATH).replace(os.sep, '/')
modelfile_content = f"""FROM {BASE_MODEL}

# Fine-tuned F1 Expert Model
# Based on google/gemma-3-4b-it with LoRA adapter
# Training: 50 examples from Formula_1_Dataset, 3 epochs

ADAPTER {adapter_dir_abs}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM You are an F1 expert assistant. You have been fine-tuned on Formula 1 racing data and can provide detailed information about drivers, teams, lap times, race positions, and Grand Prix results. Answer questions about F1 with specific data and insights.
"""

with open(MODELFILE_PATH, 'w', encoding='utf-8') as f:
    f.write(modelfile_content)

print(f"✓ Modelfile creato: {MODELFILE_PATH}")
print(f"\nContenuto:")
print("-" * 60)
print(modelfile_content)
print("-" * 60)
print()

# Deploy su Ollama
print(f"🚀 Deploy del modello su Ollama...")
print(f"   Nome: {DEPLOYED_MODEL_NAME}")
print(f"   Questo potrebbe richiedere 1-2 minuti...")
print()

try:
    result = subprocess.run(
        ["ollama", "create", DEPLOYED_MODEL_NAME, "-f", MODELFILE_PATH],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"✓ Modello deployato con successo!")
    print(f"\n{result.stdout}")
except subprocess.CalledProcessError as e:
    print(f"❌ Errore durante deploy: {e}")
    print(f"   Output: {e.stdout}")
    print(f"   Error: {e.stderr}")
    sys.exit(1)

# Verifica modello deployato
print(f"\n🔍 Verifica modello deployato...")
try:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True
    )
    if DEPLOYED_MODEL_NAME in result.stdout:
        print(f"✓ Modello '{DEPLOYED_MODEL_NAME}' presente in Ollama")
        
        # Mostra info modello
        lines = result.stdout.split('\n')
        for line in lines:
            if DEPLOYED_MODEL_NAME in line:
                print(f"\n  {line}")
    else:
        print(f"⚠️ Modello non trovato nella lista (ma deploy potrebbe essere riuscito)")
except subprocess.CalledProcessError as e:
    print(f"⚠️ Errore durante verifica: {e}")

print(f"\n{'='*60}")
print(f"  ✅ DEPLOY COMPLETATO!")
print(f"{'='*60}\n")

print(f"🎯 Prossimi passi:")
print(f"  1. Test inference:")
print(f"     python test_ollama_inference.py")
print(f"\n  2. Test manuale da terminale:")
print(f"     ollama run {DEPLOYED_MODEL_NAME}")
print(f"\n  3. Test con OllamaWrapper:")
print(f"     Usa il modello '{DEPLOYED_MODEL_NAME}' nel tuo codice")
print()

print(f"💡 Prompt di test suggeriti:")
print(f"  - Tell me about Lewis Hamilton's performance in F1")
print(f"  - What do you know about McLaren's lap times?")
print(f"  - Who finished in position 1 at Monaco Grand Prix?")
print()
