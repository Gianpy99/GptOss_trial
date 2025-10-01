"""
UI Web Semplice con Base Model + Adapter
Versione alternativa che evita problemi di dipendenze Gemma 3
"""

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
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
print("  🎨 UI WEB - F1 Expert (Base + Adapter)")
print("="*70)
print()
print(f"📦 Caricamento base model + adapter...")
print("   (Questo richiede 2-3 minuti...)")
print()

# ==============================================================================
# Carica Base Model + Adapter
# ==============================================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Carica base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)
base_model.config.pad_token_id = tokenizer.pad_token_id

# Carica adapter LoRA
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("✓ Modello caricato (base + adapter LoRA)!")
print()

# ==============================================================================
# Funzione di Chat
# ==============================================================================
def chat(message, history, temperature=0.7, max_tokens=200):
    """Gestisce la conversazione con il modello"""
    
    # Tokenizza
    inputs = tokenizer(
        message,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Genera risposta
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=0.9,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decodifica
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Rimuovi prompt
    if response.startswith(message):
        response = response[len(message):].strip()
    
    return response

# ==============================================================================
# Interfaccia Gradio
# ==============================================================================

examples = [
    ["Tell me about Lewis Hamilton's performance in F1."],
    ["What do you know about McLaren's lap times?"],
    ["Who typically finishes in position 1 at Monaco Grand Prix?"],
    ["Compare Ferrari and Mercedes performance."],
    ["What is the fastest lap time at Monza?"],
]

custom_css = """
.container { max-width: 900px; margin: auto; }
footer { display: none !important; }
"""

with gr.Blocks(css=custom_css, title="F1 Expert Chat", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🏎️ F1 Expert Assistant (Fine-tuned)
        ### Powered by Gemma 3 + LoRA Fine-tuning
        
        Chiedi informazioni su driver, team, circuiti e statistiche F1!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                show_label=True,
                avatar_images=("👤", "🤖"),
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Messaggio",
                    placeholder="Scrivi la tua domanda F1...",
                    show_label=False,
                    scale=5
                )
                submit = gr.Button("🚀 Invia", variant="primary", scale=1)
            
            with gr.Row():
                clear = gr.Button("🗑️ Cancella", size="sm")
                
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Parametri Generation")
            
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="🌡️ Temperature",
                info="0=preciso, 1=creativo"
            )
            
            max_tokens = gr.Slider(
                minimum=50,
                maximum=500,
                value=200,
                step=50,
                label="📏 Max Tokens",
                info="Lunghezza risposta"
            )
            
            gr.Markdown("---")
            gr.Markdown("### 💡 Esempi Rapidi")
            
            example_buttons = []
            for example in examples:
                btn = gr.Button(f"📌 {example[0][:30]}...", size="sm")
                example_buttons.append((btn, example[0]))
            
            gr.Markdown("---")
            gr.Markdown(
                """
                ### ℹ️ Info Modello
                - **Base**: gemma-3-4b-it
                - **Fine-tuning**: LoRA (F1 dataset)
                - **Device**: CPU
                - **Status**: ✅ Online
                """
            )
    
    # Eventi
    def respond(message, chat_history, temp, tokens):
        if not message or not message.strip():
            return chat_history, ""
        
        # Genera risposta
        bot_message = chat(message, chat_history, temp, tokens)
        
        # Aggiungi alla cronologia
        chat_history.append((message, bot_message))
        
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
    
    # Bottoni esempio
    for btn, example_text in example_buttons:
        btn.click(
            lambda x=example_text: x,
            outputs=msg
        )
    
    # Clear
    clear.click(lambda: [], outputs=chatbot, queue=False)

# ==============================================================================
# Avvio
# ==============================================================================
if __name__ == "__main__":
    print("="*70)
    print("  🚀 AVVIO UI WEB")
    print("="*70)
    print()
    print("📱 Apertura browser automatica...")
    print("🌐 URL: http://localhost:7860")
    print()
    print("⚠️  Per fermare: Ctrl+C nel terminal")
    print()
    
    demo.queue()  # Abilita code per requests multiple
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )
