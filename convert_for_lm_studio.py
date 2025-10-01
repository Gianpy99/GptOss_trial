"""
Conversione rapida in GGUF per LM Studio
Usa llama-cpp-python (più facile di llama.cpp puro)
"""

import os
import sys

print("="*70)
print("  🔄 CONVERSIONE GGUF PER LM STUDIO")
print("="*70)
print()

MERGED_MODEL = "./fine_tuned_models/f1_expert_merged"
OUTPUT_GGUF = "./fine_tuned_models/f1_expert_q4.gguf"

# Verifica che il modello esista
if not os.path.exists(MERGED_MODEL):
    print("❌ Modello merged non trovato!")
    sys.exit(1)

print("📋 Passaggi:")
print()
print("1️⃣  INSTALLA llama-cpp-python:")
print("    pip install llama-cpp-python")
print()
print("2️⃣  CONVERTI con convert-hf-to-gguf.py:")
print()
print("    Opzione A - Se hai llama.cpp clonato:")
print("    ----------------------------------------")
print("    cd path/to/llama.cpp")
print(f'    python convert-hf-to-gguf.py "{os.path.abspath(MERGED_MODEL)}" \\')
print(f'           --outfile "{os.path.abspath(OUTPUT_GGUF)}" \\')
print('           --outtype f16')
print()
print("    Opzione B - Download script standalone:")
print("    ----------------------------------------")
print("    curl -O https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert-hf-to-gguf.py")
print(f'    python convert-hf-to-gguf.py "{os.path.abspath(MERGED_MODEL)}" \\')
print(f'           --outfile "{os.path.abspath(OUTPUT_GGUF)}" \\')
print('           --outtype f16')
print()
print("3️⃣  QUANTIZZA (opzionale, riduce dimensione):")
print("    cd path/to/llama.cpp/build/bin/Release")
print(f'    .\\llama-quantize.exe "{os.path.abspath(OUTPUT_GGUF)}" \\')
print(f'                        "{os.path.abspath(OUTPUT_GGUF.replace(".gguf", "_q4_k_m.gguf"))}" \\')
print('                        Q4_K_M')
print()

print("="*70)
print()
print("⚠️  NOTA IMPORTANTE:")
print("    llama.cpp potrebbe NON supportare ancora Gemma 3!")
print("    In quel caso:")
print()
print("    1. Usa il modello direttamente con Python:")
print("       python test_inference_cpu.py")
print()
print("    2. O aspetta aggiornamento llama.cpp/LM Studio")
print()
print("    3. O usa un'interfaccia alternativa (text-generation-webui)")
print()

print("🔍 Vuoi verificare se llama.cpp supporta Gemma 3?")
print("   Vai su: https://github.com/ggerganov/llama.cpp/issues")
print("   Cerca: 'Gemma 3 support'")
print()
