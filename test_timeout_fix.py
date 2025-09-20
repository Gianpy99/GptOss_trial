#!/usr/bin/env python3
"""
Test con timeout maggiore per vedere se risolve il problema
"""

import os
import sys
sys.path.insert(0, 'src')

from ollama_wrapper import OllamaWrapper

def test_with_longer_timeout():
    print("⏱️ TEST CON TIMEOUT ESTESO")
    print("=" * 40)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    print("1. Test con timeout 90 secondi...")
    try:
        response = wrapper.chat("What is this?", files=[image_path], timeout=90)
        print(f"   Status: {response.get('status')}")
        if response.get('status') == 'success':
            answer = response.get('assistant', '')
            print(f"   🎉 FUNZIONA! Risposta: {answer[:100]}...")
        else:
            print(f"   Errore: {response.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   Eccezione: {e}")
    
    print("\n2. Test con timeout 120 secondi e prompt per Artura...")
    try:
        response = wrapper.chat(
            "This is a McLaren Artura. What can you tell me about this car?", 
            files=[image_path], 
            timeout=120
        )
        print(f"   Status: {response.get('status')}")
        if response.get('status') == 'success':
            answer = response.get('assistant', '')
            if 'artura' in answer.lower():
                print(f"   🎯 HA IMPARATO! Riconosce Artura!")
            else:
                print(f"   📝 Risposta: {answer[:150]}...")
        else:
            print(f"   Errore: {response.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   Eccezione: {e}")

if __name__ == "__main__":
    test_with_longer_timeout()