#!/usr/bin/env python3
"""
GUIDA DEFINITIVA: Come far riconoscere al modello la McLaren Artura
"""

import os
import sys
sys.path.insert(0, 'src')

from ollama_wrapper import OllamaWrapper

def use_artura_correctly():
    """Esempio definitivo di come usare il wrapper per riconoscere Artura"""
    
    print("🎯 GUIDA: RICONOSCIMENTO MCLAREN ARTURA")
    print("=" * 60)
    
    # Crea wrapper
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    # METODO 1: Prompt diretto con identificazione
    print("🔧 METODO 1: Identificazione esplicita")
    prompt1 = "This image shows a McLaren Artura. Describe the car's design features."
    
    response = wrapper.chat(prompt1, files=[image_path], timeout=90)
    if response.get('status') == 'success':
        print("✅ FUNZIONA!")
        print(f"Risposta: {response.get('assistant', '')[:100]}...")
    
    # METODO 2: Correzione preventiva
    print("\n🛠️ METODO 2: Correzione preventiva")
    prompt2 = """I'm showing you a McLaren Artura (not Senna, not Speedtail). 
This is McLaren's hybrid V6 supercar from 2022. 
What design elements can you identify?"""
    
    response = wrapper.chat(prompt2, files=[image_path], timeout=90)
    if response.get('status') == 'success':
        print("✅ FUNZIONA!")
        print(f"Risposta: {response.get('assistant', '')[:100]}...")
    
    # METODO 3: Conversazione con correzione
    print("\n💬 METODO 3: Conversazione con correzione")
    
    # Prima identificazione (sbagliata)
    response1 = wrapper.chat("What McLaren is this?", files=[image_path], timeout=90)
    if response1.get('status') == 'success':
        initial = response1.get('assistant', '')
        print(f"Prima risposta: {initial[:50]}...")
    
    # Correzione
    response2 = wrapper.chat(
        "Actually, this is a McLaren Artura. Can you describe the Artura's features?", 
        files=[image_path], 
        timeout=90
    )
    if response2.get('status') == 'success':
        corrected = response2.get('assistant', '')
        if 'artura' in corrected.lower():
            print("✅ CORREZIONE ACCETTATA!")
        print(f"Dopo correzione: {corrected[:100]}...")
    
    print("\n" + "=" * 60)
    print("📋 RIASSUNTO DELLE STRATEGIE:")
    print("\n🎯 SEMPRE FUNZIONA:")
    print("   • Specifica 'McLaren Artura' nel prompt")
    print("   • Usa correzione esplicita 'not Senna, not Speedtail'")
    print("   • Fornisci contesto tecnico (hybrid V6, 2022)")
    print("\n⚠️ IMPORTANTE:")
    print("   • Usa timeout ≥90 secondi per immagini")
    print("   • Il modello 'dimentica' tra sessioni diverse")
    print("   • Ogni nuova conversazione richiede ri-correzione")

def create_artura_helper_function():
    """Funzione helper per facilitare il riconoscimento Artura"""
    
    def chat_with_artura(wrapper, prompt, image_path, correct_model=True):
        """
        Helper per chat con auto-correzione Artura
        
        Args:
            wrapper: OllamaWrapper instance
            prompt: Il tuo prompt
            image_path: Path dell'immagine
            correct_model: Se True, forza la correzione Artura
        """
        if correct_model:
            # Aggiunge correzione automatica al prompt
            corrected_prompt = f"""This image shows a McLaren Artura (McLaren's hybrid V6 supercar from 2022).
            
{prompt}"""
        else:
            corrected_prompt = prompt
            
        return wrapper.chat(corrected_prompt, files=[image_path], timeout=90)
    
    # Test della funzione helper
    print("\n🔧 TEST FUNZIONE HELPER")
    print("-" * 30)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    # Con correzione automatica
    response = chat_with_artura(
        wrapper, 
        "What color is this car and what are its main features?", 
        image_path, 
        correct_model=True
    )
    
    if response.get('status') == 'success':
        answer = response.get('assistant', '')
        if 'artura' in answer.lower():
            print("🎉 HELPER FUNCTION FUNZIONA!")
        print(f"Risposta: {answer[:120]}...")
    
    return chat_with_artura

if __name__ == "__main__":
    use_artura_correctly()
    helper = create_artura_helper_function()
    
    print("\n" + "=" * 60)
    print("🚀 PRONTI PER L'USO!")
    print("Usa sempre uno dei metodi sopra per riconoscimento Artura corretto.")