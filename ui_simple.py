"""
UI Web Semplice per Modello Fine-tuned
Usa Gradio - interfaccia web con 3 righe di codice!

Installa: pip install gradio
Esegui: python ui_simple.py
"""

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# ==============================================================================
# Configurazione
# ==============================================================================
MODEL_PATH = "./fine_tuned_models/f1_expert_merged"  # Modello merged
USE_BASE_MODEL = False  # Se True, usa base model senza fine-tuning

if USE_BASE_MODEL:
    MODEL_PATH = "google/gemma-3-4b-it"
    
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

print("="*70)
print("  🎨 UI WEB SEMPLICE - Modello F1 Expert")
print("="*70)
print()
print(f"📦 Caricamento modello: {MODEL_PATH}")
print("   (Questo richiede 2-3 minuti...)")
print()

# ==============================================================================
# Carica Modello e Tokenizer
# ==============================================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

print("✓ Modello caricato!")
print()

# ==============================================================================
# Funzione di Chat
# ==============================================================================
def chat(message, history, temperature=0.7, max_tokens=200):
    """
    Gestisce la conversazione con il modello
    
    Args:
        message: Il messaggio dell'utente
        history: Cronologia chat (lista di tuple [user, assistant])
        temperature: Temperatura per generation (0-1)
        max_tokens: Numero massimo di token da generare
    """
    
    # Tokenizza il messaggio
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
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decodifica la risposta
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Rimuovi il prompt dall'output
    if response.startswith(message):
        response = response[len(message):].strip()
    
    return response

# ==============================================================================
# Interfaccia Gradio
# ==============================================================================

# Esempi di domande
examples = [
    ["Tell me about Lewis Hamilton's performance in F1."],
    ["What do you know about McLaren's lap times?"],
    ["Who typically finishes in position 1 at Monaco Grand Prix?"],
    ["Compare Ferrari and Mercedes performance in 2023."],
    ["What is the fastest lap time at Monza?"],
]

# Tema e CSS custom
custom_css = """
.container {
    max-width: 900px;
    margin: auto;
}
"""

# Crea interfaccia
with gr.Blocks(css=custom_css, title="F1 Expert Chat") as demo:
    
    gr.Markdown(
        """
        # 🏎️ F1 Expert Assistant
        ### Modello Fine-tuned su Dati Formula 1
        
        Chiedi informazioni su driver, team, circuiti e statistiche F1!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                show_label=True,
            )
            
            msg = gr.Textbox(
                label="Il tuo messaggio",
                placeholder="Es: Tell me about Max Verstappen...",
                show_label=False,
            )
            
            with gr.Row():
                submit = gr.Button("Invia", variant="primary")
                clear = gr.Button("Cancella Chat")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Parametri")
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Creatività (0.1=preciso, 1.0=creativo)"
            )
            
            max_tokens = gr.Slider(
                minimum=50,
                maximum=500,
                value=200,
                step=50,
                label="Max Tokens",
                info="Lunghezza risposta"
            )
            
            gr.Markdown("### 📝 Esempi")
            gr.Examples(
                examples=examples,
                inputs=msg,
            )
            
            gr.Markdown(
                f"""
                ### ℹ️ Info
                - **Modello**: {MODEL_PATH.split('/')[-1]}
                - **Device**: CPU
                - **Status**: ✅ Pronto
                """
            )
    
    # Eventi
    def respond(message, chat_history, temp, tokens):
        if not message.strip():
            return chat_history, ""
        
        # Genera risposta
        bot_message = chat(message, chat_history, temp, tokens)
        
        # Aggiungi alla cronologia
        chat_history.append((message, bot_message))
        
        return chat_history, ""
    
    # Submit con Enter o bottone
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
    
    clear.click(lambda: None, None, chatbot, queue=False)

# ==============================================================================
# Avvia l'interfaccia
# ==============================================================================
if __name__ == "__main__":
    print("="*70)
    print("  🚀 AVVIO UI WEB")
    print("="*70)
    print()
    print("📱 L'interfaccia si aprirà automaticamente nel browser")
    print("🌐 URL locale: http://localhost:7860")
    print()
    print("💡 Per fermare: Ctrl+C nel terminal")
    print()
    
    demo.launch(
        server_name="127.0.0.1",  # Solo locale
        server_port=7860,
        share=False,  # NON condividere online
        inbrowser=True,  # Apri browser automaticamente
    )
