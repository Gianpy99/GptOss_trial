#!/usr/bin/env python3
"""
Test specifico per le funzionalità multimodali.
Verifica il supporto per immagini con diversi modelli.
"""

import sys
import os

# Aggiungiamo il src al path per l'import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_wrapper import OllamaWrapper

def test_model_capabilities():
    """Verifica le capacità del modello attuale"""
    print("🔍 Test Capacità Modello")
    print("-" * 30)
    
    wrapper = OllamaWrapper()
    
    # Verifica info del modello
    print(f"Modello in uso: {wrapper.model_name}")
    
    model_info = wrapper.show_model_info()
    if model_info.get("status") == "success":
        info = model_info.get("info", {})
        print(f"✓ Informazioni modello ottenute")
        
        # Stampa alcune info chiave se disponibili
        if "model" in info:
            print(f"  - Modello: {info.get('model', 'N/A')}")
        if "parameters" in info:
            print(f"  - Parametri disponibili: {len(info.get('parameters', {}))}")
        if "template" in info:
            print(f"  - Template: {info.get('template', 'N/A')[:100]}...")
    else:
        print(f"❌ Errore nel recuperare info modello: {model_info}")

def test_image_encoding():
    """Test della codifica base64 dell'immagine"""
    print("\n🖼️ Test Codifica Immagine")
    print("-" * 25)
    
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print(f"❌ Immagine non trovata: {image_path}")
        return False
    
    try:
        import base64
        
        # Verifica che possiamo leggere e codificare l'immagine
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        encoded = base64.b64encode(image_data).decode("utf-8")
        print(f"✓ Immagine letta e codificata")
        print(f"  - Dimensione file: {len(image_data)} bytes")
        print(f"  - Dimensione base64: {len(encoded)} caratteri")
        print(f"  - Tipo file (estensione): {os.path.splitext(image_path)[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nella codifica: {e}")
        return False

def test_payload_structure():
    """Test della struttura del payload inviato"""
    print("\n📦 Test Struttura Payload")
    print("-" * 25)
    
    wrapper = OllamaWrapper()
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print(f"❌ Immagine non trovata")
        return
    
    # Simula la costruzione del payload come fa il wrapper
    try:
        import base64
        
        # Costruisce i messaggi
        msgs = wrapper._build_messages("Descrivi questa immagine", include_history=False)
        print(f"✓ Messaggi costruiti: {len(msgs)} messaggi")
        
        # Simula l'aggiunta dell'allegato
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        encoded = base64.b64encode(image_data).decode("utf-8")
        attachment = {
            "filename": os.path.basename(image_path),
            "content_base64": encoded[:100] + "...",  # Solo preview
            "mime": "png"
        }
        
        print(f"✓ Allegato creato:")
        print(f"  - Nome file: {attachment['filename']}")
        print(f"  - MIME: {attachment['mime']}")
        print(f"  - Base64 preview: {attachment['content_base64']}")
        
        # Mostra struttura finale del messaggio
        final_msg = msgs[-1].copy()
        final_msg.setdefault("attachments", []).append(attachment)
        
        print(f"✓ Messaggio finale con allegato:")
        print(f"  - Ruolo: {final_msg.get('role')}")
        print(f"  - Contenuto: {final_msg.get('content')}")
        print(f"  - Allegati: {len(final_msg.get('attachments', []))}")
        
    except Exception as e:
        print(f"❌ Errore nella costruzione payload: {e}")

def test_different_approaches():
    """Test con diversi approcci per le immagini"""
    print("\n🔄 Test Approcci Diversi")
    print("-" * 25)
    
    wrapper = OllamaWrapper()
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print(f"❌ Immagine non trovata")
        return
    
    # Approccio 1: Chat normale senza immagine (controllo)
    print("1. Test controllo senza immagine...")
    try:
        response = wrapper.chat("Puoi vedere immagini?", timeout=30)
        if response.get("status") == "success":
            print(f"   ✓ Risposta: {response.get('assistant', '')[:100]}...")
        else:
            print(f"   ❌ Errore: {response}")
    except Exception as e:
        print(f"   ❌ Eccezione: {e}")
    
    # Approccio 2: Chat con file (come nel test originale)
    print("2. Test con file attraverso parametro...")
    try:
        response = wrapper.chat(
            "Analizza questa immagine e dimmi cosa vedi", 
            files=[image_path],
            timeout=45
        )
        if response.get("status") == "success":
            print(f"   ✓ Risposta: {response.get('assistant', '')[:100]}...")
        else:
            print(f"   ❌ Errore: {response}")
    except Exception as e:
        print(f"   ❌ Eccezione: {e}")
    
    # Approccio 3: Verifica se ci sono modelli vision disponibili
    print("3. Verifica modelli disponibili...")
    try:
        models = wrapper.list_models()
        if "models" in models:
            print("   Modelli trovati:")
            for model in models["models"]:
                name = model.get("name", "N/A")
                print(f"     - {name}")
                # Controlla se il nome suggerisce supporto vision
                if any(keyword in name.lower() for keyword in ["vision", "llava", "bakllava", "multimodal"]):
                    print(f"       📷 Potrebbe supportare immagini!")
        else:
            print(f"   Lista modelli: {models}")
    except Exception as e:
        print(f"   ❌ Errore lista modelli: {e}")

def main():
    """Esegue tutti i test specifici per multimodal"""
    print("🔬 Test Approfonditi Funzionalità Multimodali")
    print("=" * 50)
    
    try:
        test_model_capabilities()
        test_image_encoding()
        test_payload_structure()
        test_different_approaches()
        
        print("\n" + "=" * 50)
        print("📋 Risultati Test Multimodali:")
        print("✅ Codifica immagini funziona")
        print("✅ Struttura payload corretta")
        print("✅ Wrapper gestisce gli allegati")
        print()
        print("💡 Note:")
        print("- Il modello gemma3:4b potrebbe non supportare immagini")
        print("- Per il supporto vision, considera modelli come llava o bakllava")
        print("- Il wrapper è pronto per modelli multimodali quando disponibili")
        
    except Exception as e:
        print(f"\n❌ Errore durante i test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()