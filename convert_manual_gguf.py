"""
Conversione manuale GGUF bypassando transformers
Usiamo solo safetensors + gguf writer diretto
"""
import struct
import json
import numpy as np
from pathlib import Path
from safetensors import safe_open
import gguf

# Paths
MODEL_DIR = Path("C:/Development/Ollama_wrapper/fine_tuned_models/f1_expert_merged")
OUTPUT_FILE = Path("C:/Development/Ollama_wrapper/fine_tuned_models/f1_expert.gguf")

print("🔄 Conversione GGUF Manuale (Bypass Transformers)")
print(f"📁 Input: {MODEL_DIR}")
print(f"💾 Output: {OUTPUT_FILE}")
print()

# Step 1: Leggi config.json
config_path = MODEL_DIR / "config.json"
print(f"1️⃣ Carico config.json...")
with open(config_path) as f:
    config = json.load(f)

print(f"   ✓ Model: {config.get('model_type', 'unknown')}")
print(f"   ✓ Hidden size: {config.get('hidden_size')}")
print(f"   ✓ Layers: {config.get('num_hidden_layers')}")
print()

# Step 2: Inizializza GGUF writer
print("2️⃣ Inizializzo writer GGUF...")
writer = gguf.GGUFWriter(str(OUTPUT_FILE), arch="gemma3")

# Metadata essenziale
writer.add_name("f1_expert_merged")
writer.add_description("Fine-tuned F1 Expert Model")
writer.add_file_type(gguf.GGMLQuantizationType.F16)

# Architettura (da config.json)
writer.add_block_count(config.get("num_hidden_layers", 24))
writer.add_context_length(config.get("max_position_embeddings", 8192))
writer.add_embedding_length(config.get("hidden_size", 3072))
writer.add_head_count(config.get("num_attention_heads", 24))
writer.add_head_count_kv(config.get("num_key_value_heads", 8))
writer.add_layer_norm_rms_eps(config.get("rms_norm_eps", 1e-6))

# Vocabulary size
vocab_size = config.get("vocab_size", 256000)
writer.add_vocab_size(vocab_size)

print(f"   ✓ Metadata aggiunto")
print()

# Step 3: Tokenizer (sempificato)
print("3️⃣ Aggiungo tokenizer...")
tokenizer_path = MODEL_DIR / "tokenizer.json"
if tokenizer_path.exists():
    with open(tokenizer_path, encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    # Estrai vocab dal tokenizer
    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    
    # Aggiungi tokens
    tokens = []
    scores = []
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        tokens.append(token.encode("utf-8"))
        scores.append(0.0)  # Score placeholder
    
    writer.add_tokenizer_model("llama")  # Gemma usa Llama tokenizer
    writer.add_token_list(tokens[:vocab_size])
    writer.add_token_scores(scores[:vocab_size])
    
    print(f"   ✓ {len(tokens[:vocab_size])} tokens aggiunti")
else:
    print("   ⚠️ tokenizer.json non trovato, skip")

print()

# Step 4: Converti tensori da safetensors
print("4️⃣ Converto tensori (può richiedere 5-10 minuti)...")

# Trova tutti i file safetensors
safetensor_files = sorted(MODEL_DIR.glob("model-*.safetensors"))
print(f"   📦 {len(safetensor_files)} file safetensors trovati")
print()

tensor_count = 0
for shard_idx, shard_path in enumerate(safetensor_files, 1):
    print(f"   🔄 Shard {shard_idx}/{len(safetensor_files)}: {shard_path.name}")
    
    with safe_open(shard_path, framework="numpy", device="cpu") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            
            # Converti a float16
            if tensor.dtype != np.float16:
                tensor = tensor.astype(np.float16)
            
            # Aggiungi al GGUF
            writer.add_tensor(name, tensor)
            tensor_count += 1
            
            if tensor_count % 50 == 0:
                print(f"      ✓ {tensor_count} tensori convertiti...")

print()
print(f"   ✅ Totale: {tensor_count} tensori convertiti")
print()

# Step 5: Scrivi file finale
print("5️⃣ Scrivo file GGUF...")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

file_size_gb = OUTPUT_FILE.stat().st_size / (1024**3)
print()
print("=" * 60)
print(f"✅ CONVERSIONE COMPLETATA!")
print(f"📄 File: {OUTPUT_FILE}")
print(f"💾 Dimensione: {file_size_gb:.2f} GB")
print("=" * 60)
print()
print("🚀 Prossimi step:")
print("   1. Quantizza: python convert_quantize.py")
print("   2. Test: llama-cli -m f1_expert.gguf")
print("   3. Import Ollama: ollama create f1-expert -f Modelfile-gguf")
