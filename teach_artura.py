#!/usr/bin/env python3
"""
Test definitivo per insegnare al modello McLaren Artura
"""

import os
import sys
sys.path.insert(0, 'src')

from ollama_wrapper import OllamaWrapper

def mclaren_artura_training():
    print("🏎️ TRAINING MCLAREN ARTURA - Versione Definitiva")
    print("=" * 60)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    # Session dedicata per l'apprendimento Artura
    wrapper.session_id = "artura_learning"
    
    print("📚 FASE 1: Correzione iniziale")
    print("-" * 30)
    
    # Step 1: Identificazione errata
    print("1. Cosa vede il modello inizialmente...")
    response = wrapper.chat("What McLaren model is this?", files=[image_path], timeout=90)
    if response.get('status') == 'success':
        initial_answer = response.get('assistant', '')
        print(f"   Risposta iniziale: {initial_answer[:100]}...")
        if 'senna' in initial_answer.lower():
            print("   ❌ Identifica erroneamente come Senna")
        elif 'speedtail' in initial_answer.lower():
            print("   ❌ Identifica erroneamente come Speedtail")
        else:
            print("   ❓ Identificazione incerta")
    
    # Step 2: Correzione diretta e ferma
    print("\n2. Correzione diretta...")
    correction_prompt = """No, this is actually a McLaren Artura, not a Senna. 
The McLaren Artura is McLaren's hybrid V6 supercar that was launched in 2022.
It features a twin-turbo V6 hybrid engine and represents McLaren's new generation of supercars.
Please acknowledge this correction and describe what you can see in this McLaren Artura."""
    
    response = wrapper.chat(correction_prompt, files=[image_path], timeout=120)
    if response.get('status') == 'success':
        correction_answer = response.get('assistant', '')
        print(f"   Risposta dopo correzione: {correction_answer[:150]}...")
        if 'artura' in correction_answer.lower():
            print("   ✅ Ha accettato la correzione!")
        else:
            print("   ⚠️ Non ha riconosciuto esplicitamente Artura")
    
    # Step 3: Rinforzo con dettagli tecnici
    print("\n3. Rinforzo con specifiche...")
    tech_prompt = """Perfect! This McLaren Artura has these key characteristics:
- Twin-turbo V6 hybrid engine (first hybrid McLaren since P1)
- Carbon fiber monocoque
- Distinctive LED headlight design
- Launch year: 2022
- It's McLaren's "entry-level" supercar but still incredibly advanced
What design elements can you identify in this blue McLaren Artura?"""
    
    response = wrapper.chat(tech_prompt, files=[image_path], timeout=120)
    if response.get('status') == 'success':
        tech_answer = response.get('assistant', '')
        print(f"   Risposta tecnica: {tech_answer[:150]}...")
    
    print("\n" + "=" * 60)
    print("🧪 FASE 2: Test di verifica apprendimento")
    print("-" * 40)
    
    # Test 1: Verifica memoria
    print("1. Test memoria modello...")
    response = wrapper.chat("What model did we just discuss?", timeout=60)
    if response.get('status') == 'success':
        memory_answer = response.get('assistant', '')
        if 'artura' in memory_answer.lower():
            print("   🧠 MEMORIA OK! Ricorda che è Artura")
        else:
            print("   ⚠️ Non ricorda esplicitamente Artura")
        print(f"   Risposta: {memory_answer[:100]}...")
    
    # Test 2: Identificazione senza suggerimenti
    print("\n2. Test identificazione pura (nuova conversazione)...")
    # Nuova sessione per test pulito
    wrapper_test = OllamaWrapper(model_name="gemma3:4b")
    wrapper_test.session_id = "clean_test"
    
    response = wrapper_test.chat("What car is this?", files=[image_path], timeout=90)
    if response.get('status') == 'success':
        clean_answer = response.get('assistant', '')
        if 'artura' in clean_answer.lower():
            print("   🎉 APPRENDIMENTO PERMANENTE! Riconosce Artura senza aiuto!")
        elif 'senna' in clean_answer.lower():
            print("   📚 Ancora confuso con Senna - serve più training")
        else:
            print("   ❓ Risposta generica")
        print(f"   Risposta pulita: {clean_answer[:120]}...")
    
    # Test 3: Prompt di verifica specifica
    print("\n3. Test con prompt di verifica...")
    verification_prompt = "Is this a McLaren Artura? Please explain your reasoning."
    response = wrapper.chat(verification_prompt, files=[image_path], timeout=90)
    if response.get('status') == 'success':
        verify_answer = response.get('assistant', '')
        if 'yes' in verify_answer.lower() and 'artura' in verify_answer.lower():
            print("   ✅ CONFERMA POSITIVA! È sicuro che sia Artura")
        elif 'artura' in verify_answer.lower():
            print("   📝 Menziona Artura ma non è certo")
        else:
            print("   ❌ Non conferma Artura")
        print(f"   Verifica: {verify_answer[:120]}...")
    
    # Salva la sessione di training
    save_result = wrapper.save_session("artura_trained_model")
    if save_result.get('status') == 'success':
        print(f"\n💾 Sessione training salvata: {save_result.get('file')}")
    
    print("\n" + "=" * 60)
    print("📋 RISULTATI TRAINING:")
    print(f"✅ Correzione diretta applicata")
    print(f"✅ Specifiche tecniche fornite") 
    print(f"✅ Test di verifica completati")
    print(f"💡 Per training persistente:")
    print(f"   - Carica sempre sessione 'artura_trained_model'")
    print(f"   - Usa timeout ≥90s per immagini")
    print(f"   - Correggi immediatamente identificazioni errate")

def test_trained_model():
    """Test rapido del modello trainato"""
    print("\n🔬 TEST RAPIDO MODELLO TRAINATO")
    print("=" * 40)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    
    # Carica la sessione di training
    load_result = wrapper.load_session("artura_trained_model")
    if load_result.get('status') == 'success':
        print("✅ Sessione training caricata")
        
        image_path = os.path.join("examples", "_2ARTURA_Blue.png")
        response = wrapper.chat("Describe this car", files=[image_path], timeout=90)
        
        if response.get('status') == 'success':
            answer = response.get('assistant', '')
            if 'artura' in answer.lower():
                print("🎯 PERFETTO! Modello trainato riconosce Artura!")
            else:
                print("📚 Serve ancora training...")
            print(f"Risposta: {answer[:150]}...")
    else:
        print("❌ Sessione training non trovata")

if __name__ == "__main__":
    mclaren_artura_training()
    test_trained_model()