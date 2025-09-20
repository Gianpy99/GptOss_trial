#!/usr/bin/env python3
"""
Test rapido per verificare che la correzione del formato immagini funzioni.
"""

import sys
import os

# Aggiungiamo il src al path per l'import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_wrapper import OllamaWrapper

def test_fixed_vision():
    """Test del wrapper corretto per le immagini"""
    print("🎯 Test Wrapper Corretto - Vision")
    print("=" * 35)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print(f"❌ Immagine non trovata: {image_path}")
        return
    
    print(f"📷 Testing con: {image_path}")
    
    # Test 1: Descrizione generale
    print("\n1. Test descrizione...")
    try:
        response = wrapper.chat(
            "Descrivi questa immagine in dettaglio", 
            files=[image_path],
            timeout=60
        )
        
        if response.get("status") == "success":
            description = response.get("assistant", "")
            print(f"✅ VISION FUNZIONA!")
            print(f"Descrizione: {description[:200]}...")
            
            # Controlla se menciona caratteristiche dell'auto
            if any(word in description.lower() for word in ["mclaren", "blu", "blue", "car", "auto", "supercar"]):
                print("🎉 Il modello vede correttamente l'auto McLaren!")
            else:
                print("ℹ️ Descrizione generica")
                
        else:
            print(f"❌ Errore: {response}")
            
    except Exception as e:
        print(f"❌ Eccezione: {e}")
    
    # Test 2: Domanda specifica
    print("\n2. Test domanda specifica...")
    try:
        response = wrapper.chat(
            "Che marca di auto è questa?", 
            files=[image_path],
            timeout=45
        )
        
        if response.get("status") == "success":
            answer = response.get("assistant", "")
            print(f"✅ Risposta: {answer[:150]}...")
            
            if "mclaren" in answer.lower():
                print("🎯 PERFETTO! Ha identificato correttamente la McLaren!")
            else:
                print("ℹ️ Risposta data, ma potrebbe non aver identificato la marca")
                
        else:
            print(f"❌ Errore: {response}")
            
    except Exception as e:
        print(f"❌ Eccezione: {e}")
    
    # Test 3: Colori
    print("\n3. Test colori...")
    try:
        response = wrapper.chat(
            "Che colore è quest'auto?", 
            files=[image_path],
            timeout=30
        )
        
        if response.get("status") == "success":
            color_answer = response.get("assistant", "")
            print(f"✅ Colori: {color_answer[:100]}...")
            
            if any(color in color_answer.lower() for color in ["blu", "blue", "azzurr"]):
                print("🎨 Ha identificato correttamente il colore blu!")
                
        else:
            print(f"❌ Errore: {response}")
            
    except Exception as e:
        print(f"❌ Eccezione: {e}")

def main():
    """Esegue il test della correzione"""
    try:
        test_fixed_vision()
        
        print("\n" + "=" * 35)
        print("🎉 TEST VISION COMPLETATO!")
        print("\nSe vedi descrizioni corrette dell'auto McLaren blu,")
        print("allora il wrapper è stato corretto con successo! 🚗💙")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()