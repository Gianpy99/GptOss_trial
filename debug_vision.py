#!/usr/bin/env python3
"""
Test debug per capire perché gemma3:4b non risponde alle immagini
nonostante abbia la capability "vision" in Ollama.
"""

import sys
import os
import json
import base64

# Aggiungiamo il src al path per l'import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_wrapper import OllamaWrapper

def debug_payload_structure():
    """Debug della struttura del payload inviato"""
    print("🔍 Debug Payload Struttura")
    print("-" * 30)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print(f"❌ Immagine non trovata: {image_path}")
        return
    
    # Costruisce manualmente il payload per vedere cosa stiamo inviando
    msgs = wrapper._build_messages("Descrivi questa immagine", include_history=False)
    
    # Aggiungi l'immagine come fa il wrapper
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    encoded = base64.b64encode(image_data).decode("utf-8")
    
    # Mostra il payload che stiamo costruendo
    print(f"✓ Messaggi di base: {len(msgs)}")
    print(f"✓ Ultimo messaggio: {msgs[-1]}")
    
    # Come il wrapper aggiunge l'allegato
    attachment = {
        "filename": os.path.basename(image_path),
        "content_base64": encoded[:100] + "...[troncato]",
        "mime": "png"
    }
    
    final_msg = msgs[-1].copy()
    final_msg.setdefault("attachments", []).append(attachment)
    
    print(f"✓ Messaggio finale con attachment:")
    print(f"  - Role: {final_msg.get('role')}")
    print(f"  - Content: {final_msg.get('content')}")
    print(f"  - Ha attachments: {'attachments' in final_msg}")
    print(f"  - Numero attachments: {len(final_msg.get('attachments', []))}")

def test_ollama_api_directly():
    """Test diretto delle API Ollama per le immagini"""
    print(f"\n🔗 Test API Ollama Diretta")
    print("-" * 30)
    
    import requests
    
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print(f"❌ Immagine non trovata")
        return
    
    # Codifica l'immagine
    with open(image_path, "rb") as f:
        image_data = f.read()
    encoded = base64.b64encode(image_data).decode("utf-8")
    
    # Test 1: Formato standard con images
    print("1. Test formato con 'images' array...")
    payload1 = {
        "model": "gemma3:4b",
        "messages": [
            {
                "role": "user",
                "content": "Descrivi questa immagine",
                "images": [encoded]
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload1,
            timeout=60
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if "message" in result:
                content = result["message"].get("content", "")
                print(f"   ✅ Risposta: {content[:150]}...")
            else:
                print(f"   Struttura risposta: {result}")
        else:
            print(f"   ❌ Errore: {response.text}")
    except Exception as e:
        print(f"   ❌ Eccezione: {e}")
    
    # Test 2: Formato con attachments (come fa il nostro wrapper)
    print("\n2. Test formato con 'attachments'...")
    payload2 = {
        "model": "gemma3:4b",
        "messages": [
            {
                "role": "user",
                "content": "Descrivi questa immagine",
                "attachments": [
                    {
                        "filename": "_2ARTURA_Blue.png",
                        "content_base64": encoded,
                        "mime": "png"
                    }
                ]
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload2,
            timeout=60
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if "message" in result:
                content = result["message"].get("content", "")
                print(f"   ✅ Risposta: {content[:150]}...")
            else:
                print(f"   Struttura risposta: {result}")
        else:
            print(f"   ❌ Errore: {response.text}")
    except Exception as e:
        print(f"   ❌ Eccezione: {e}")

def test_different_prompts():
    """Test con diversi tipi di prompt"""
    print(f"\n💬 Test Prompts Diversi")
    print("-" * 25)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print(f"❌ Immagine non trovata")
        return
    
    prompts = [
        "What do you see in this image?",
        "Describe the image",
        "Analizza l'immagine",
        "Tell me about the colors in this picture",
        "<image>What is this?",
        "I'm uploading an image. What does it show?"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        try:
            # Usa direttamente l'API con il formato 'images'
            import requests
            
            with open(image_path, "rb") as f:
                image_data = f.read()
            encoded = base64.b64encode(image_data).decode("utf-8")
            
            payload = {
                "model": "gemma3:4b",
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt,
                        "images": [encoded]
                    }
                ],
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                print(f"   ✅ {content[:100]}...")
                
                # Controlla se la risposta sembra vision-aware
                vision_words = ["blue", "blu", "color", "colore", "image", "immagine", "see", "vedo"]
                if any(word in content.lower() for word in vision_words):
                    print(f"   🎉 POSSIBILE RISPOSTA VISION!")
            else:
                print(f"   ❌ Errore HTTP: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Eccezione: {e}")

def main():
    """Esegue tutti i test di debug"""
    print("🐛 Debug Gemma3:4b Vision Support")
    print("=" * 50)
    
    try:
        debug_payload_structure()
        test_ollama_api_directly()
        test_different_prompts()
        
        print("\n" + "=" * 50)
        print("📋 Risultati Debug:")
        print("✅ Ollama conferma che gemma3:4b ha capability 'vision'")
        print("✅ Il modello su HuggingFace supporta immagini")
        print("🔍 Controlliamo se il formato del payload è corretto")
        print("🔍 Testiamo diversi formati di invio immagini")
        print()
        print("💡 Se i test sopra mostrano risposte vision-aware,")
        print("   allora il problema è nel nostro wrapper wrapper.py")
        print("   e dobbiamo aggiornare il formato degli allegati.")
        
    except Exception as e:
        print(f"\n❌ Errore durante debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()