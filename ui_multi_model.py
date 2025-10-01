"""
🌐 Ollama Multi-Model Web UI
UI Gradio con selector di modelli dinamico
"""

import gradio as gr
import requests
import subprocess
from typing import List, Tuple

# Configurazione
OLLAMA_API = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "f1-expert"

def get_available_models() -> List[str]:
    """Recupera lista modelli da Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = [line.split()[0] for line in lines if line.strip()]
            return models if models else [DEFAULT_MODEL]
        
    except Exception as e:
        print(f"Error fetching models: {e}")
    
    return [DEFAULT_MODEL]

def chat_with_model(message: str, history: List[Tuple[str, str]], model_name: str, 
                    temperature: float, max_tokens: int) -> Tuple[str, List[Tuple[str, str]]]:
    """Invia messaggio al modello e ritorna risposta."""
    
    if not message.strip():
        return "", history
    
    try:
        # Prepara il prompt con lo storico
        full_prompt = ""
        for user_msg, assistant_msg in history:
            full_prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        full_prompt += f"User: {message}\nAssistant:"
        
        # Chiamata API Ollama
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model_name,
                "prompt": message,  # Ollama gestisce automaticamente il context
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": max_tokens
                }
            },
            timeout=120
        )
        
        response.raise_for_status()
        result = response.json()
        
        assistant_response = result.get("response", "")
        
        # Aggiorna history
        history.append((message, assistant_response))
        
        return "", history
        
    except requests.exceptions.Timeout:
        error_msg = "⏱️ Request timeout - il modello sta impiegando troppo tempo"
        history.append((message, error_msg))
        return "", history
        
    except requests.exceptions.ConnectionError:
        error_msg = "🔌 Errore di connessione - Ollama non è in esecuzione?\n\nAvvia Ollama con: `ollama serve`"
        history.append((message, error_msg))
        return "", history
        
    except Exception as e:
        error_msg = f"❌ Errore: {str(e)}"
        history.append((message, error_msg))
        return "", history

def clear_chat():
    """Pulisce la chat."""
    return []

def get_model_info(model_name: str) -> str:
    """Recupera informazioni sul modello."""
    try:
        result = subprocess.run(
            ["ollama", "show", model_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Estrai info base
            output = result.stdout
            info = f"**Model**: {model_name}\n\n"
            
            # Cerca parametri
            if "Parameters" in output or "parameters" in output:
                lines = output.split('\n')
                for i, line in enumerate(lines):
                    if 'parameter' in line.lower() or 'temperature' in line.lower():
                        info += f"- {line.strip()}\n"
            
            return info if len(info) > 50 else f"**Model**: {model_name}\n\nNo additional info available."
        
    except Exception as e:
        return f"**Model**: {model_name}\n\n⚠️ Could not fetch model info: {e}"
    
    return f"**Model**: {model_name}"

# Crea interfaccia Gradio
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="🏎️ Ollama Multi-Model Chat",
    css="""
        .gradio-container {max-width: 1200px !important}
        #chatbot {min-height: 500px}
        #model-info {font-size: 0.9em; padding: 10px; background: #f5f5f5; border-radius: 5px}
    """
) as app:
    
    # Header
    gr.Markdown("""
    # 🏎️ Ollama Multi-Model Chat
    **Chat con i tuoi modelli fine-tuned** - Supporta F1, Tolkien, e modelli ibridi
    """)
    
    with gr.Row():
        # Colonna principale: chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="💬 Conversation",
                height=500,
                show_copy_button=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Ask about Formula 1, Tolkien, or anything else...",
                    scale=4,
                    show_label=False
                )
                submit_btn = gr.Button("Send 🚀", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                example_f1 = gr.Button("💡 F1 Example", size="sm")
                example_tolkien = gr.Button("💡 Tolkien Example", size="sm")
        
        # Colonna laterale: settings
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Model Settings")
            
            model_dropdown = gr.Dropdown(
                choices=get_available_models(),
                value=DEFAULT_MODEL,
                label="Select Model",
                info="Choose which model to chat with"
            )
            
            refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
            
            model_info = gr.Markdown(
                get_model_info(DEFAULT_MODEL),
                elem_id="model-info"
            )
            
            gr.Markdown("### 🎛️ Generation Parameters")
            
            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Higher = more creative"
            )
            
            max_tokens = gr.Slider(
                minimum=50,
                maximum=500,
                value=250,
                step=50,
                label="Max Tokens",
                info="Maximum response length"
            )
            
            gr.Markdown("""
            ---
            **Quick Tips:**
            - 🔄 Refresh models after training
            - 🌡️ Lower temp = precise
            - 🌡️ Higher temp = creative
            """)
    
    # Eventi
    def send_message(message, history, model, temp, tokens):
        return chat_with_model(message, history, model, temp, tokens)
    
    def refresh_models():
        models = get_available_models()
        return gr.Dropdown(choices=models, value=models[0] if models else DEFAULT_MODEL)
    
    def update_model_info(model_name):
        return get_model_info(model_name)
    
    # Submit message
    submit_btn.click(
        fn=send_message,
        inputs=[msg, chatbot, model_dropdown, temperature, max_tokens],
        outputs=[msg, chatbot]
    )
    
    msg.submit(
        fn=send_message,
        inputs=[msg, chatbot, model_dropdown, temperature, max_tokens],
        outputs=[msg, chatbot]
    )
    
    # Clear chat
    clear_btn.click(fn=clear_chat, outputs=[chatbot])
    
    # Refresh models
    refresh_btn.click(fn=refresh_models, outputs=[model_dropdown])
    
    # Update model info quando cambia modello
    model_dropdown.change(fn=update_model_info, inputs=[model_dropdown], outputs=[model_info])
    
    # Example buttons
    example_f1.click(
        fn=lambda h, m, t, tok: chat_with_model("Who won the 2008 Formula 1 championship?", h, m, t, tok),
        inputs=[chatbot, model_dropdown, temperature, max_tokens],
        outputs=[msg, chatbot]
    )
    
    example_tolkien.click(
        fn=lambda h, m, t, tok: chat_with_model("Who is Gandalf in The Lord of the Rings?", h, m, t, tok),
        inputs=[chatbot, model_dropdown, temperature, max_tokens],
        outputs=[msg, chatbot]
    )

if __name__ == "__main__":
    print("🌐 Starting Ollama Multi-Model UI...")
    print("📊 Detecting available models...")
    
    models = get_available_models()
    print(f"✅ Found {len(models)} model(s): {', '.join(models)}")
    
    print("\n🚀 Launching UI on http://localhost:7860")
    print("💡 Tip: Use 'ollama-cli list --type models' to see all models\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
