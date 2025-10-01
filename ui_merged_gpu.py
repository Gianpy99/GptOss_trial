"""
UI GPU con Modello MERGED
Usa il modello già merged invece di base + adapter
Evita problemi di caricamento LoRA su GPU
"""

import os
import sys

print("="*70)
print("  🏎️ F1 Expert UI - Modello Merged GPU")
print("="*70)
print()

print("📦 Step 1/4: Import libraries...")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv

print("✓ Libraries imported")
print()

# Config - USA MODELLO MERGED!
MODEL_PATH = "./fine_tuned_models/f1_expert_merged"  # Modello completo
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Check GPU
print("📦 Step 2/4: Check GPU...")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  No GPU found, will use CPU (slow)")
print()

# Tokenizer
print("📦 Step 3/4: Load tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)

# FIX pad_token
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 0:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"✓ Tokenizer loaded (pad_token_id fixed: 0 → {tokenizer.pad_token_id})")
else:
    print(f"✓ Tokenizer loaded (pad_token_id={tokenizer.pad_token_id})")
print()

# Model MERGED con quantizzazione
print("📦 Step 4/4: Load merged model (30-60 seconds)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)

# Sync config
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.eval()

print("✓ Merged model loaded on GPU")
print()

# Chat function
def chat(message, history, temperature=0.7, max_tokens=150):
    """Generate response on GPU"""
    if not message or not message.strip():
        return ""
    
    try:
        # Tokenize
        inputs = tokenizer(
            message,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        ).to(model.device)
        
        # Generate
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
                use_cache=True
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(message):
            response = response[len(message):].strip()
        
        return response
        
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"

# Gradio UI
print("="*70)
print("  🚀 STARTING WEB UI")
print("="*70)
print()

import gradio as gr

examples = [
    ["Tell me about Lewis Hamilton's performance in F1."],
    ["What do you know about McLaren's lap times?"],
    ["Who typically finishes in position 1 at Monaco Grand Prix?"],
]

with gr.Blocks(title="F1 Expert Merged", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏎️ F1 Expert Assistant (GPU - Merged Model)
        ### Gemma 3 Fine-tuned (Base + LoRA merged)
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=450,
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about F1...",
                    show_label=False,
                    scale=5
                )
                submit = gr.Button("🚀", variant="primary", scale=1)
            
            clear = gr.ClearButton([msg, chatbot], value="Clear")
                
        with gr.Column(scale=1):
            temperature = gr.Slider(0.1, 1.0, 0.7, step=0.1, label="Temperature")
            max_tokens = gr.Slider(50, 300, 150, step=25, label="Max Length")
            
            gr.Examples(examples=examples, inputs=msg)
            
            gr.Markdown(
                f"""
                **Model**: Merged (Base + LoRA)  
                **Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}  
                **Quant**: 4-bit NF4  
                **Status**: ✅ Ready
                """
            )
    
    def respond(message, chat_history, temp, tokens):
        if not message or not message.strip():
            return chat_history, ""
        
        bot_message = chat(message, chat_history, temp, tokens)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        
        return chat_history, ""
    
    msg.submit(respond, [msg, chatbot, temperature, max_tokens], [chatbot, msg])
    submit.click(respond, [msg, chatbot, temperature, max_tokens], [chatbot, msg])

print("📱 Opening browser at http://localhost:7862")
print("⚡ GPU ready for fast inference!")
print()
print("💡 Usando modello MERGED (evita problemi LoRA)")
print()
print("⚠️  Press Ctrl+C to stop")
print()

demo.queue()
demo.launch(
    server_name="127.0.0.1",
    server_port=7862,
    share=False,
    inbrowser=True,
    show_error=True
)
