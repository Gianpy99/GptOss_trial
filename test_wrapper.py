#!/usr/bin/env python3
"""
Test rapido del wrapper Ollama per verificare che funzioni correttamente.
"""

import sys
import os

# Aggiungiamo il src al path per l'import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from ollama_wrapper import OllamaWrapper
    print("✓ Import riuscito")
    
    # Test connessione e lista modelli
    print("\n=== Test connessione Ollama ===")
    wrapper = OllamaWrapper()
    
    print(f"Base URL: {wrapper.base_url}")
    print(f"Modello: {wrapper.model_name}")
    
    # Lista modelli disponibili
    print("\n=== Lista modelli disponibili ===")
    models = wrapper.list_models()
    if "error" in models:
        print(f"❌ Errore nel recuperare i modelli: {models['error']}")
    else:
        print("✓ Modelli disponibili:")
        if "models" in models:
            for model in models["models"][:3]:  # Mostra solo i primi 3
                print(f"  - {model.get('name', 'N/A')}")
        else:
            print(f"  {models}")
    
    # Test semplice chat
    print("\n=== Test chat semplice ===")
    try:
        response = wrapper.chat("Ciao, puoi rispondere brevemente in italiano?", timeout=30)
        if response.get("status") == "success":
            print("✓ Chat funziona!")
            print(f"Risposta: {response.get('assistant', 'N/A')[:100]}...")
        else:
            print(f"❌ Errore nella chat: {response}")
    except Exception as e:
        print(f"❌ Eccezione durante la chat: {e}")
    
    print("\n=== Test completato ===")
    
except ImportError as e:
    print(f"❌ Errore import: {e}")
except Exception as e:
    print(f"❌ Errore generico: {e}")