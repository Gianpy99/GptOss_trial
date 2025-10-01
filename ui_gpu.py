"""
UI Gradio Ottimizzata per GPU
Versione leggera per ambiente inference con quantizzazione 4-bit
"""

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

# ==============================================================================
# Configurazione
# ==============================================================================
BASE_MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_PATH = "./finetuning_projects/f1_expert_fixed/adapter"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

print("="*70)
print("  🎨 UI WEB GPU - F1 Expert")
print("="*70)
print()
print(f"🎮 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()
print(f"📦 Caricamento modello con quantizzazione 4-bit...")
print("   (30-60 secondi...)")
print()

# ==============================================================================
# Carica Model con 4-bit quantization + FIX CUDA assertion
# ==============================================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)

# FIX CRITICO per CUDA assertion: 
# Gemma ha bug con pad_token_id=0, forziamo a usare un token valido
# Gemma-3 tipicamente: eos_token_id=1 o 106
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 0:
    # Usa EOS token per padding (safe per Gemma)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"🔧 FIX: pad_token_id cambiato da 0 → {tokenizer.pad_token_id} (EOS)")
else:
    print(f"✓ pad_token_id già valido: {tokenizer.pad_token_id}")

print()

# Quantizzazione 4-bit per GPU 6GB
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Carica base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)

# IMPORTANTE: Allinea model config con tokenizer
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id

# Carica adapter LoRA
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("✓ Modello caricato su GPU con pad_token fix!")
print()

# ==============================================================================
# Funzione di Chat (GPU-optimized)
# ==============================================================================
def chat(message, history, temperature=0.7, max_tokens=200):
    """Genera risposta usando GPU"""
    
    if not message or not message.strip():
        return ""
    
    try:
        # Tokenizza con parametri sicuri
        inputs = tokenizer(
            message,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        ).to(model.device)
        
        # IMPORTANTE: Verifica che attention_mask sia corretto
        # (evita token 0 nelle posizioni padding)
        if inputs["attention_mask"].sum() == 0:
            return "❌ Errore: Input vuoto dopo tokenizzazione"
        
        # Genera su GPU con parametri sicuri
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=0.9,
                do_sample=True if temperature > 0.1 else False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                # FIX: Non permettere generazione di pad_token
                bad_words_ids=[[tokenizer.pad_token_id]] if tokenizer.pad_token_id == 0 else None
            )
        
        # Decodifica
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Rimuovi prompt
        if response.startswith(message):
            response = response[len(message):].strip()
        
        return response
        
    except Exception as e:
        return f"❌ Errore: {str(e)}"

# ==============================================================================
# Interfaccia Gradio
# ==============================================================================

examples = [
    ["Tell me about Lewis Hamilton's performance in F1."],
    ["What do you know about McLaren's lap times?"],
    ["Who typically finishes in position 1 at Monaco Grand Prix?"],
    ["Compare Ferrari and Mercedes performance."],
]

with gr.Blocks(title="F1 Expert GPU", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🏎️ F1 Expert Assistant (GPU Accelerated)
        ### Gemma 3 + LoRA Fine-tuning | 4-bit Quantization
        
        Chiedi informazioni su driver, team, circuiti e statistiche F1!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="💬 Chat",
                height=450,
                avatar_images=("👤", "🤖"),
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Es: Tell me about Max Verstappen...",
                    show_label=False,
                    scale=5,
                    container=False
                )
                submit = gr.Button("🚀", variant="primary", scale=1, min_width=50)
            
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot], value="🗑️ Cancella")
                
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="🌡️ Temperature"
            )
            
            max_tokens = gr.Slider(
                minimum=50,
                maximum=300,
                value=150,
                step=25,
                label="📏 Max Length"
            )
            
            gr.Markdown("### 💡 Quick Examples")
            gr.Examples(
                examples=examples,
                inputs=msg,
                label=None
            )
            
            gr.Markdown(
                f"""
                ### ℹ️ System Info
                - **Device**: {'🎮 GPU' if torch.cuda.is_available() else '💻 CPU'}
                - **Quantization**: 4-bit NF4
                - **Adapter**: LoRA (F1)
                - **Status**: ✅ Ready
                """
            )
    
    # Eventi
    def respond(message, chat_history, temp, tokens):
        if not message or not message.strip():
            return chat_history, ""
        
        # Genera risposta
        bot_message = chat(message, chat_history, temp, tokens)
        
        # Formato messages (nuovo Gradio)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        
        return chat_history, ""
    
    # Submit
    msg.submit(
        respond,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[chatbot, msg]
    )
    
    submit.click(
        respond,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[chatbot, msg]
    )

# ==============================================================================
# Avvio
# ==============================================================================
if __name__ == "__main__":
    print("="*70)
    print("  🚀 AVVIO UI WEB")
    print("="*70)
    print()
    print("📱 Browser: http://localhost:7860")
    print("⚡ GPU ready for fast inference!")
    print()
    print("⚠️  Ctrl+C per fermare")
    print()
    
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )
