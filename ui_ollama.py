"""
UI Gradio con Ollama Backend (GPU via Ollama)
Bypassa completamente transformers - usa Ollama API!
"""
import gradio as gr
import requests
import json

# Configurazione
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "f1-expert"

print("=" * 60)
print("🏎️ F1 Expert UI - Ollama GPU Backend")
print("=" * 60)
print()
print(f"🔗 Ollama API: {OLLAMA_API}")
print(f"🤖 Model: {MODEL_NAME}")
print()

def chat_f1(message, history):
    """Funzione chat per Gradio usando Ollama API"""
    
    try:
        # Chiamata API Ollama
        response = requests.post(
            OLLAMA_API,
            json={
                "model": MODEL_NAME,
                "prompt": message,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 250
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Errore: risposta vuota")
        else:
            return f"❌ Errore API: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "⏱️ Timeout: richiesta troppo lunga"
    except requests.exceptions.ConnectionError:
        return "❌ Errore: Ollama non in esecuzione. Avvia con `ollama serve`"
    except Exception as e:
        return f"❌ Errore: {str(e)}"

# UI Gradio
demo = gr.ChatInterface(
    chat_f1,
    title="🏎️ F1 Expert - GPU Accelerated (via Ollama)",
    description="""
    Fine-tuned Gemma 3 4B model con **GPU tramite Ollama**!
    
    ⚡ **Performance**: 1-5 secondi per risposta  
    🎯 **Modello**: Fine-tuned su Formula_1_Dataset  
    🚀 **GPU**: NVIDIA GTX 1660 SUPER (via Ollama)  
    ✅ **Nessun problema transformers**: Ollama gestisce tutto!
    """,
    examples=[
        "Tell me about Lewis Hamilton's performance in Formula 1",
        "What are McLaren's best lap times?",
        "Who won the Monaco Grand Prix?",
        "Compare Ferrari and Red Bull performance",
        "Explain DRS system in Formula 1",
        "Tell me about Max Verstappen's championships"
    ],
    theme="soft",
    cache_examples=False
)

if __name__ == "__main__":
    print("🌐 Avvio server Gradio...")
    print("📍 URL: http://localhost:7860")
    print()
    print("💡 Assicurati che Ollama sia in esecuzione!")
    print("   Se non funziona, avvia: ollama serve")
    print()
    
    demo.launch(
        server_port=7860,
        server_name="0.0.0.0",
        share=False,
        show_error=True
    )
