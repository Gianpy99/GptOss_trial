"""
Converti modello merged in formato GGUF per Ollama
Richiede: llama-cpp-python o llama.cpp

PRIMA esegui: merge_adapter.py
"""

import os
import subprocess

# ==============================================================================
# Configurazione
# ==============================================================================
MERGED_MODEL_PATH = "./fine_tuned_models/f1_expert_merged"
GGUF_OUTPUT_PATH = "./fine_tuned_models/f1_expert.gguf"

print("="*70)
print("  🔄 CONVERSIONE IN FORMATO GGUF PER OLLAMA")
print("="*70)
print()

# Verifica che il modello merged esista
if not os.path.exists(MERGED_MODEL_PATH):
    print("❌ ERRORE: Modello merged non trovato!")
    print(f"   Path: {MERGED_MODEL_PATH}")
    print()
    print("💡 Prima esegui: python merge_adapter.py")
    exit(1)

print("⚙️ Configurazione:")
print(f"  Input: {MERGED_MODEL_PATH}")
print(f"  Output: {GGUF_OUTPUT_PATH}")
print()

# ==============================================================================
# Metodo 1: llama-cpp-python (Più facile su Windows)
# ==============================================================================
print("📦 Metodo 1: llama-cpp-python")
print()
print("Installazione:")
print("  pip install llama-cpp-python")
print()
print("Conversione:")
print(f'  python -m llama_cpp.convert "{MERGED_MODEL_PATH}" \\')
print(f'         --outfile "{GGUF_OUTPUT_PATH}" \\')
print('         --outtype f16')
print()

# ==============================================================================
# Metodo 2: llama.cpp (Richiede build)
# ==============================================================================
print("📦 Metodo 2: llama.cpp (più controllo)")
print()
print("Setup:")
print("  1. git clone https://github.com/ggerganov/llama.cpp")
print("  2. cd llama.cpp")
print("  3. cmake -B build && cmake --build build --config Release")
print()
print("Conversione:")
print(f'  python convert-hf-to-gguf.py "{MERGED_MODEL_PATH}" \\')
print(f'         --outfile "{GGUF_OUTPUT_PATH}" \\')
print('         --outtype f16')
print()

# ==============================================================================
# Metodo 3: HuggingFace Hub (online)
# ==============================================================================
print("📦 Metodo 3: HuggingFace Hub (più semplice)")
print()
print("1. Carica il modello merged su HuggingFace:")
print("   huggingface-cli login")
print("   huggingface-cli upload tuousername/f1-expert ./fine_tuned_models/f1_expert_merged")
print()
print("2. Usa un servizio di conversione online:")
print("   https://huggingface.co/spaces/ggml-org/gguf-my-repo")
print()

# ==============================================================================
# Dopo la conversione
# ==============================================================================
print("="*70)
print("  📝 DOPO LA CONVERSIONE")
print("="*70)
print()
print("Per caricare in Ollama:")
print()
print("1. Crea Modelfile:")
print("   ---")
print(f'   FROM {GGUF_OUTPUT_PATH}')
print('   PARAMETER temperature 0.7')
print('   PARAMETER top_p 0.9')
print('   SYSTEM "You are an F1 expert assistant."')
print("   ---")
print()
print("2. Importa in Ollama:")
print("   ollama create f1-expert -f Modelfile")
print()
print("3. Testa:")
print('   ollama run f1-expert "Tell me about Lewis Hamilton"')
print()

print("💡 RACCOMANDAZIONE:")
print("   Se hai problemi con GGUF su Windows, usa LM Studio che")
print("   può caricare direttamente i safetensors del modello merged!")
print()
