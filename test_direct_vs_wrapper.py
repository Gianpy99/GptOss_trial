#!/usr/bin/env python3
"""
Test diretto con i prompt che funzionavano nel debug.
"""

import sys
import os

# Aggiungiamo il src al path per l'import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_wrapper import OllamaWrapper

def test_working_prompts():
    """Test con i prompt che funzionavano nel debug"""
    print("🔥 Test con Prompt che Funzionavano")
    print("=" * 40)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print(f"❌ Immagine non trovata: {image_path}")
        return
    
    # I prompt che funzionavano nel debug
    working_prompts = [
        "Describe the image",
        "Tell me about the colors in this picture", 
        "<image>What is this?",
        "I'm uploading an image. What does it show?"
    ]
    
    for i, prompt in enumerate(working_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        try:
            response = wrapper.chat(
                prompt,
                files=[image_path],
                timeout=60
            )
            
            if response.get("status") == "success":
                answer = response.get("assistant", "")
                print(f"   ✅ Risposta: {answer[:150]}...")
                
                # Controlla se menziona caratteristiche dell'auto
                vision_keywords = ["mclaren", "blue", "car", "supercar", "vehicle", "automotive"]
                if any(keyword in answer.lower() for keyword in vision_keywords):
                    print(f"   🎯 VISION ATTIVA! Ha riconosciuto l'auto!")
                else:
                    print(f"   ❓ Risposta ma senza riconoscimento auto")
                    
            else:
                print(f"   ❌ Errore: {response}")
                
        except Exception as e:
            print(f"   ❌ Eccezione: {e}")

def test_wrapper_vs_direct():
    """Confronta wrapper vs chiamata diretta"""
    print(f"\n⚖️ Confronto Wrapper vs Diretto")
    print("-" * 35)
    
    import requests
    import base64
    
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print("❌ Immagine non trovata")
        return
    
    # Test con chiamata diretta (che funzionava)
    print("1. Chiamata API diretta...")
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        encoded = base64.b64encode(image_data).decode("utf-8")
        
        payload = {
            "model": "gemma3:4b",
            "messages": [
                {
                    "role": "user",
                    "content": "Describe the image",
                    "images": [encoded]
                }
            ],
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("message", {}).get("content", "")
            print(f"   ✅ API diretta: {content[:100]}...")
        else:
            print(f"   ❌ Errore API: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Eccezione API: {e}")
    
    # Test con wrapper
    print("\n2. Wrapper...")
    try:
        wrapper = OllamaWrapper(model_name="gemma3:4b")
        response = wrapper.chat(
            "Describe the image",
            files=[image_path],
            timeout=60
        )
        
        if response.get("status") == "success":
            content = response.get("assistant", "")
            print(f"   ✅ Wrapper: {content[:100]}...")
        else:
            print(f"   ❌ Errore wrapper: {response}")
            
    except Exception as e:
        print(f"   ❌ Eccezione wrapper: {e}")

def main():
    """Esegue tutti i test"""
    try:
        test_working_prompts()
        test_wrapper_vs_direct()
        
        print("\n" + "=" * 40)
        print("🔍 Analisi:")
        print("Se l'API diretta funziona ma il wrapper no,")
        print("c'è ancora qualcosa da correggere nel formato.")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()