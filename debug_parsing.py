#!/usr/bin/env python3
"""
Debug del problema parsing response vuote
"""

import os
import sys
sys.path.insert(0, 'src')

from ollama_wrapper import OllamaWrapper

def debug_response_parsing():
    print("🔍 DEBUG RESPONSE PARSING")
    print("=" * 40)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    print("1. Test senza immagine...")
    try:
        response = wrapper.chat("Ciao, come stai?", timeout=30)
        print(f"   Risposta senza immagine: {response}")
    except Exception as e:
        print(f"   Errore: {e}")
    
    print("\n2. Test con immagine - prompt semplice...")
    try:
        response = wrapper.chat("What is this?", files=[image_path], timeout=45)
        print(f"   Risposta con immagine: {response}")
        print(f"   Tipo response: {type(response)}")
        if isinstance(response, dict):
            print(f"   Keys: {response.keys()}")
            for key, value in response.items():
                print(f"   {key}: {str(value)[:100]}...")
    except Exception as e:
        print(f"   Errore: {e}")
    
    print("\n3. Test debug interno wrapper...")
    # Aggiungiamo debug diretto al wrapper per capire cosa succede
    import requests
    import base64
    import json
    
    # Prepara immagine
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Payload diretto
    payload = {
        "model": "gemma3:4b",
        "messages": [
            {
                "role": "user",
                "content": "What is this?",
                "images": [image_data]
            }
        ],
        "stream": False
    }
    
    print("   Chiamata API diretta...")
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=45
        )
        print(f"   Status: {response.status_code}")
        print(f"   Raw response: {response.text[:200]}...")
        
        if response.status_code == 200:
            json_response = response.json()
            print(f"   JSON keys: {json_response.keys()}")
            if 'message' in json_response:
                message = json_response['message']
                print(f"   Message: {message}")
                if 'content' in message:
                    print(f"   Content: {message['content'][:100]}...")
    except Exception as e:
        print(f"   Errore API diretta: {e}")

if __name__ == "__main__":
    debug_response_parsing()