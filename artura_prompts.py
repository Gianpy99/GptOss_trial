#!/usr/bin/env python3
"""
Template di prompt ottimali per McLaren Artura
"""

import os
import sys
sys.path.insert(0, 'src')

from ollama_wrapper import OllamaWrapper

# Template di prompt che funzionano per identificare correttamente l'Artura
ARTURA_PROMPTS = {
    "identification": "This image shows a McLaren Artura. What can you tell me about this car?",
    
    "correction": "This is a McLaren Artura, not a Senna or Speedtail. The Artura is McLaren's V6 hybrid supercar from 2022. Describe what you see.",
    
    "technical": "Analyze this McLaren Artura's design features. The Artura has a twin-turbo V6 hybrid engine and carbon fiber construction.",
    
    "specific": "Looking at this blue McLaren Artura hybrid supercar, what design elements can you identify?",
    
    "comparison": "This McLaren Artura is different from the Senna or Speedtail. What unique Artura features do you notice?",
}

def test_optimal_prompts():
    print("🎯 TEST PROMPT OTTIMALI PER ARTURA")
    print("=" * 50)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    for name, prompt in ARTURA_PROMPTS.items():
        print(f"\n📝 {name.upper()}: '{prompt[:60]}...'")
        
        try:
            response = wrapper.chat(prompt, files=[image_path], timeout=90)
            if response.get('status') == 'success':
                answer = response.get('assistant', '')
                
                # Analisi della risposta
                if 'artura' in answer.lower():
                    print("   🎯 ✅ RICONOSCE ARTURA!")
                elif 'senna' in answer.lower():
                    print("   ❌ Dice ancora Senna")
                elif 'speedtail' in answer.lower():
                    print("   ❌ Dice ancora Speedtail")
                else:
                    print("   ℹ️ Risposta generica")
                
                print(f"   Risposta: {answer[:120]}...")
            else:
                print(f"   ❌ Errore: {response.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"   ❌ Eccezione: {e}")

def create_artura_assistant():
    """Crea un assistente pre-configurato per riconoscere Artura"""
    print("\n🤖 CREAZIONE ASSISTENTE ARTURA")
    print("=" * 40)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    wrapper.session_id = "artura_expert"
    
    # System prompt specializzato
    artura_system_prompt = """You are an expert on McLaren supercars, specifically the McLaren Artura.

IMPORTANT: When analyzing images of blue McLaren supercars, remember:
- The McLaren Artura is a hybrid V6 supercar launched in 2022
- It has distinctive LED headlights and aerodynamic design
- It's NOT a Senna (track-focused) or Speedtail (streamlined)
- The Artura represents McLaren's new generation of hybrid technology

Always identify McLaren Artura correctly when you see the blue car in images."""
    
    wrapper.set_system_prompt(artura_system_prompt)
    
    # Test l'assistente
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    response = wrapper.chat("What car is this?", files=[image_path], timeout=90)
    
    if response.get('status') == 'success':
        answer = response.get('assistant', '')
        if 'artura' in answer.lower():
            print("🎉 ASSISTENTE ARTURA CREATO CON SUCCESSO!")
        else:
            print("⚠️ Assistente creato ma serve raffinamento")
        print(f"Risposta assistente: {answer[:150]}...")
        
        # Salva l'assistente
        wrapper.save_session("artura_expert")
        print("💾 Assistente salvato come 'artura_expert'")
    else:
        print(f"❌ Errore creazione assistente: {response.get('error', 'Unknown')}")

if __name__ == "__main__":
    test_optimal_prompts()
    create_artura_assistant()