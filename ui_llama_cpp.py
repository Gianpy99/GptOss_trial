"""
UI Gradio con llama-cpp-python (GPU via GGUF)
Bypassa completamente transformers - nessun import loop!
"""
import gradio as gr
from llama_cpp import Llama
import torch

# Configurazione
MODEL_PATH = "C:/Development/Ollama_wrapper/fine_tuned_models/f1_expert.gguf"
N_GPU_LAYERS = 35  # Carica 35 layer su GPU (GTX 1660 SUPER 6GB)
N_CTX = 2048       # Context window
MAX_TOKENS = 250   # Max token risposta

print("=" * 60)
print("🚀 F1 Expert UI - llama.cpp GPU")
print("=" * 60)
print()

# Verifica GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_memory:.1f} GB")
else:
    print("⚠️ CUDA non disponibile, uso CPU")
    N_GPU_LAYERS = 0

print()
print(f"📁 Carico modello: {MODEL_PATH}")
print(f"🎮 GPU Layers: {N_GPU_LAYERS}")
print(f"⏳ Caricamento in corso (30-60 secondi)...")
print()

# Carica modello GGUF
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=N_GPU_LAYERS,
    n_ctx=N_CTX,
    n_threads=8,
    verbose=False
)

print("✅ Modello caricato!")
print()
print("=" * 60)
print()

def chat_f1(message, history):
    """Funzione chat per Gradio"""
    
    # Build prompt con history
    prompt = "<|im_start|>system\nYou are an expert Formula 1 assistant. Provide detailed information about F1 drivers, teams, circuits, and statistics.<|im_end|>\n"
    
    for user_msg, bot_msg in history:
        prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{bot_msg}<|im_end|>\n"
    
    prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    
    # Genera risposta
    response = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        stop=["<|im_end|>", "<|endoftext|>"],
        echo=False
    )
    
    return response["choices"][0]["text"].strip()

# UI Gradio
demo = gr.ChatInterface(
    chat_f1,
    title="🏎️ F1 Expert - GPU Accelerated (GGUF)",
    description="""
    Fine-tuned model usando **llama.cpp** con **CUDA GPU**!
    
    ⚡ **Performance**: 1-3 secondi per risposta  
    🎯 **Modello**: Gemma 3 4B fine-tuned su F1 Dataset  
    🚀 **GPU**: NVIDIA GTX 1660 SUPER (6GB)
    """,
    examples=[
        "Tell me about Lewis Hamilton's performance in Formula 1",
        "What are McLaren's best lap times?",
        "Who won the Monaco Grand Prix last year?",
        "Compare Ferrari and Red Bull performance",
        "Explain DRS system in Formula 1"
    ],
    theme="soft",
    cache_examples=False
)

if __name__ == "__main__":
    print("🌐 Avvio server Gradio...")
    print("📍 URL: http://localhost:7865")
    print()
    demo.launch(
        server_port=7865,
        server_name="0.0.0.0",
        share=False
    )
