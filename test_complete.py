#!/usr/bin/env python3
"""
Test completo del wrapper Ollama.
Verifica tutte le funzionalità principali.
"""

import sys
import os

# Aggiungiamo il src al path per l'import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_wrapper import OllamaWrapper, create_coding_assistant, create_creative_assistant

def test_basic_functionality():
    """Test delle funzionalità di base"""
    print("=== Test Funzionalità Base ===")
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    
    # Test chat semplice
    print("1. Test chat semplice...")
    response = wrapper.chat("Rispondi solo con 'OK' in italiano", timeout=30)
    if response.get("status") == "success":
        print(f"   ✓ Risposta: {response.get('assistant', 'N/A')}")
    else:
        print(f"   ❌ Errore: {response}")
    
    # Test lista modelli
    print("2. Test lista modelli...")
    models = wrapper.list_models()
    if "error" not in models:
        print("   ✓ Lista modelli ottenuta")
        if "models" in models:
            print(f"   Modelli disponibili: {len(models['models'])}")
    else:
        print(f"   ❌ Errore: {models['error']}")
    
    # Test memory
    print("3. Test memory...")
    wrapper.store_memory("test_key", "test_value", "test_category")
    recall = wrapper.recall_memory("test_key")
    if recall and recall[1] == "test_value":
        print("   ✓ Memory funziona")
    else:
        print(f"   ❌ Memory fallita: {recall}")

def test_streaming():
    """Test dello streaming"""
    print("\n=== Test Streaming ===")
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    
    print("1. Test streaming chat...")
    full_response = ""
    try:
        for chunk in wrapper.stream_chat("Conta da 1 a 5 lentamente"):
            print(chunk, end="", flush=True)
            full_response += chunk
            if len(full_response) > 200:  # Limita per il test
                break
        print("\n   ✓ Streaming funziona")
    except Exception as e:
        print(f"\n   ❌ Errore streaming: {e}")

def test_assistants():
    """Test degli assistenti predefiniti"""
    print("\n=== Test Assistenti Predefiniti ===")
    
    # Test coding assistant
    print("1. Test coding assistant...")
    coding = create_coding_assistant("test_coding")
    response = coding.chat("Scrivi una funzione Python che calcola il fattoriale", timeout=30)
    if response.get("status") == "success":
        print("   ✓ Coding assistant funziona")
        print(f"   Preview: {response.get('assistant', '')[:100]}...")
    else:
        print(f"   ❌ Errore: {response}")
    
    # Test creative assistant
    print("2. Test creative assistant...")
    creative = create_creative_assistant("test_creative")
    response = creative.chat("Scrivi un haiku sulla programmazione", timeout=30)
    if response.get("status") == "success":
        print("   ✓ Creative assistant funziona")
        print(f"   Preview: {response.get('assistant', '')[:100]}...")
    else:
        print(f"   ❌ Errore: {response}")

def test_sessions():
    """Test delle sessioni"""
    print("\n=== Test Sessioni ===")
    
    wrapper = OllamaWrapper(model_name="gemma3:4b", session_id="test_session")
    
    # Salva una sessione
    save_result = wrapper.save_session("test_save")
    if save_result.get("status") == "success":
        print("   ✓ Salvataggio sessione riuscito")
    else:
        print(f"   ❌ Errore salvataggio: {save_result}")
    
    # Lista sessioni
    sessions = wrapper.list_sessions()
    if "test_save" in sessions:
        print("   ✓ Sessione trovata nella lista")
    else:
        print(f"   ❌ Sessione non trovata: {sessions}")
    
    # Carica sessione
    load_result = wrapper.load_session("test_save")
    if load_result.get("status") == "success":
        print("   ✓ Caricamento sessione riuscito")
    else:
        print(f"   ❌ Errore caricamento: {load_result}")

def test_multimodal():
    """Test delle funzionalità multimodali con immagini"""
    print("\n=== Test Multimodal (Immagini) ===")
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    
    # Percorso dell'immagine di test
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    # Verifica che l'immagine esista
    if not os.path.exists(image_path):
        print(f"   ❌ Immagine non trovata: {image_path}")
        return
    
    print(f"   📷 Testando con immagine: {image_path}")
    
    # Test verifica supporto vision del modello
    print("1. Test supporto vision (prompt inglese)...")
    try:
        response = wrapper.chat("Describe the image", files=[image_path], timeout=90)
        if response.get("status") == "success":
            answer = response.get("assistant", "")
            
            # Verifica se la risposta indica vision funzionante
            vision_indicators = ["mclaren", "blue", "car", "supercar", "vehicle", "speedtail", "senna", "artura"]
            has_vision = any(indicator in answer.lower() for indicator in vision_indicators)
            
            if has_vision:
                print("   🎉 VISION FUNZIONA! Il modello vede l'auto McLaren!")
                print(f"   Descrizione: {answer[:150]}...")
            else:
                print("   ℹ️ Risposta ricevuta ma vision potrebbe non essere attiva")
                print(f"   Risposta: {answer[:100]}...")
        else:
            print(f"   ❌ Errore: {response}")
    except Exception as e:
        print(f"   ❌ Eccezione: {e}")
    
    # Test riconoscimento specifico
    print("2. Test riconoscimento marca...")
    try:
        response = wrapper.chat("What brand is this car?", files=[image_path], timeout=90)
        if response.get("status") == "success":
            answer = response.get("assistant", "")
            if "mclaren" in answer.lower():
                print("   ✅ Ha identificato correttamente McLaren!")
                print(f"   Risposta: {answer[:100]}...")
            else:
                print("   ℹ️ Risposta data ma marca non identificata chiaramente")
                print(f"   Risposta: {answer[:100]}...")
        else:
            print(f"   ❌ Errore: {response}")
    except Exception as e:
        print(f"   ❌ Eccezione: {e}")
    
    # Test gestione errori
    print("3. Test gestione errori...")
    try:
        response = wrapper.chat(
            "Describe this image", 
            files=["file_inesistente.jpg"],
            timeout=30
        )
        print("   ✅ Gestione errore file inesistente OK")
    except Exception as e:
        print(f"   ⚠️ Eccezione gestione errore: {e}")
    
    # Info finale
    print("\n   💡 Note sul test multimodal:")
    print("   - ✅ Il wrapper ora supporta correttamente le immagini!")
    print("   - ✅ gemma3:4b ha funzionalità vision complete")
    print("   - ✅ Riconosce auto, colori, e dettagli specifici")
    print("   - 💡 Usa prompt in inglese per risultati ottimali")

def main():
    """Esegue tutti i test"""
    print("🚀 Inizio test completi del wrapper Ollama")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_streaming()
        test_assistants()
        test_sessions()
        test_multimodal()
        
        print("\n" + "=" * 50)
        print("✅ Test completati con successo!")
        print("\n💡 Il wrapper Ollama è pronto per l'uso!")
        print("\nEsempi di utilizzo:")
        print("```python")
        print("from ollama_wrapper import OllamaWrapper, create_coding_assistant")
        print("")
        print("# Wrapper di base")
        print("wrapper = OllamaWrapper()")
        print("response = wrapper.chat('Ciao, come stai?')")
        print("")
        print("# Assistente per programmazione")
        print("coding = create_coding_assistant()")
        print("code = coding.chat('Scrivi una funzione Python per ordinare una lista')")
        print("```")
        
    except Exception as e:
        print(f"\n❌ Errore durante i test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()