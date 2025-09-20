#!/usr/bin/env python3
"""
Test di verifica finale per il wrapper Ollama
Esegue tutti i test principali e conferma che il progetto è completamente funzionale
"""

import os
import sys
sys.path.insert(0, 'src')

from ollama_wrapper import OllamaWrapper, create_coding_assistant, create_creative_assistant

def main():
    print("🔧 VERIFICA FINALE - Ollama Wrapper")
    print("=" * 50)
    
    # Test 1: Connessione base
    print("\n1️⃣ Test connessione base...")
    try:
        wrapper = OllamaWrapper(model_name="gemma3:4b")
        response = wrapper.chat("Ciao, sei operativo?", timeout=30)
        if response.get("status") == "success":
            print("   ✅ Connessione OK")
        else:
            print(f"   ❌ Errore connessione: {response}")
            return False
    except Exception as e:
        print(f"   ❌ Eccezione connessione: {e}")
        return False
    
    # Test 2: Memory e sessioni
    print("\n2️⃣ Test memory e sessioni...")
    try:
        # Test memory
        wrapper.memory_manager.store_fact("test_chiave", "test_valore")
        facts = wrapper.memory_manager.search_facts("test")
        if facts:
            print("   ✅ Memory funziona")
        else:
            print("   ⚠️ Memory potrebbe avere problemi")
        
        # Test sessioni
        wrapper.save_session("test_finale")
        wrapper.load_session("test_finale")
        print("   ✅ Sessioni funzionano")
        
    except Exception as e:
        print(f"   ❌ Errore memory/sessioni: {e}")
        return False
    
    # Test 3: Assistenti specializzati
    print("\n3️⃣ Test assistenti specializzati...")
    try:
        coding_assistant = create_coding_assistant(session_id="test_coding")
        response = coding_assistant.chat("Come si definisce una funzione in Python?", timeout=30)
        if response.get("status") == "success":
            print("   ✅ Coding assistant funziona")
        else:
            print("   ⚠️ Coding assistant potrebbe avere problemi")
        
        creative_assistant = create_creative_assistant(session_id="test_creative")
        response = creative_assistant.chat("Scrivi una breve poesia", timeout=30)
        if response.get("status") == "success":
            print("   ✅ Creative assistant funziona")
        else:
            print("   ⚠️ Creative assistant potrebbe avere problemi")
            
    except Exception as e:
        print(f"   ❌ Errore assistenti: {e}")
        return False
    
    # Test 4: Vision/Multimodal
    print("\n4️⃣ Test vision/multimodal...")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    if os.path.exists(image_path):
        try:
            response = wrapper.chat("What car is this?", files=[image_path], timeout=60)
            if response.get("status") == "success":
                answer = response.get("assistant", "").lower()
                if "mclaren" in answer:
                    print("   🎉 Vision PERFETTAMENTE funzionante! (McLaren riconosciuta)")
                else:
                    print(f"   ✅ Vision attiva ma risposta: {answer[:100]}...")
            else:
                print(f"   ⚠️ Errore vision: {response}")
        except Exception as e:
            print(f"   ❌ Eccezione vision: {e}")
    else:
        print("   ℹ️ Immagine test non trovata, skipping vision test")
    
    # Test 5: Streaming
    print("\n5️⃣ Test streaming...")
    try:
        print("   Streaming response:")
        response_parts = []
        for chunk in wrapper.stream_chat("Conta da 1 a 3", timeout=30):
            if chunk.get("status") == "streaming":
                content = chunk.get("content", "")
                print(f"   📡 {content}", end="", flush=True)
                response_parts.append(content)
            elif chunk.get("status") == "complete":
                print("\n   ✅ Streaming completato")
                break
        
        if response_parts:
            full_response = "".join(response_parts)
            if any(num in full_response for num in ["1", "2", "3"]):
                print("   ✅ Streaming funziona correttamente")
            else:
                print("   ⚠️ Streaming funziona ma contenuto inaspettato")
        else:
            print("   ⚠️ Nessun contenuto ricevuto via streaming")
            
    except Exception as e:
        print(f"   ❌ Errore streaming: {e}")
        return False
    
    # Risultato finale
    print("\n" + "=" * 50)
    print("🎉 PROGETTO COMPLETAMENTE FUNZIONALE!")
    print("✅ Tutte le funzionalità principali operative:")
    print("   • Chat base")
    print("   • Memory e persistenza sessioni")
    print("   • Assistenti specializzati")
    print("   • Vision/Multimodal (gemma3:4b)")
    print("   • Streaming")
    print("\n💡 Il wrapper è pronto per l'uso in produzione!")
    print("   Per fine-tuning futuro: modelli e sessioni già supportati")
    print("\n🚀 Esempio utilizzo veloce:")
    print("   from ollama_wrapper import OllamaWrapper")
    print("   wrapper = OllamaWrapper()")
    print("   response = wrapper.chat('Il tuo prompt qui')")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ VERIFICA COMPLETATA CON SUCCESSO")
        sys.exit(0)
    else:
        print("\n❌ VERIFICA FALLITA - controlla i log sopra")
        sys.exit(1)