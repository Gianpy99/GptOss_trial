#!/usr/bin/env python3
"""
Test per insegnare al modello a riconoscere correttamente la McLaren Artura
Diverse strategie di prompting e correzione
"""

import os
import sys
sys.path.insert(0, 'src')

from ollama_wrapper import OllamaWrapper

def test_artura_recognition():
    print("🏎️ TEST RICONOSCIMENTO MCLAREN ARTURA")
    print("=" * 50)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print(f"❌ Immagine non trovata: {image_path}")
        return
    
    # Strategia 1: Prompt con contesto specifico
    print("\n1️⃣ Strategia: Prompt con contesto Artura")
    prompts_with_context = [
        "This is a McLaren Artura. Describe what you see in this image.",
        "I'm showing you a McLaren Artura hybrid supercar. What details can you identify?",
        "This blue car is a McLaren Artura from 2022. Analyze its design features."
    ]
    
    for i, prompt in enumerate(prompts_with_context, 1):
        print(f"\n   Test 1.{i}: '{prompt[:50]}...'")
        try:
            response = wrapper.chat(prompt, files=[image_path], timeout=45)
            if response.get("status") == "success":
                answer = response.get("assistant", "")
                if "artura" in answer.lower():
                    print("   🎯 HA IMPARATO! Riconosce Artura!")
                elif any(model in answer.lower() for model in ["speedtail", "senna"]):
                    print("   ⚠️ Ancora confuso con altri modelli")
                else:
                    print("   ℹ️ Risposta generica")
                print(f"   Risposta: {answer[:100]}...")
            else:
                print(f"   ❌ Errore: {response}")
        except Exception as e:
            print(f"   ❌ Eccezione: {e}")
    
    # Strategia 2: Correzione progressiva
    print("\n\n2️⃣ Strategia: Correzione progressiva")
    conversation_steps = [
        ("What McLaren model is this?", "Identificazione iniziale"),
        ("No, this is actually a McLaren Artura, not a Speedtail. Can you describe the Artura's specific features?", "Correzione diretta"),
        ("The McLaren Artura is a hybrid V6 supercar from 2022. What hybrid features might be visible?", "Rinforzo informazione")
    ]
    
    for i, (prompt, description) in enumerate(conversation_steps, 1):
        print(f"\n   Step 2.{i}: {description}")
        print(f"   Prompt: '{prompt[:60]}...'")
        try:
            response = wrapper.chat(prompt, files=[image_path], timeout=45)
            if response.get("status") == "success":
                answer = response.get("assistant", "")
                if "artura" in answer.lower():
                    print("   🎉 SUCCESSO! Ha accettato la correzione!")
                else:
                    print("   📝 Continua l'apprendimento...")
                print(f"   Risposta: {answer[:120]}...")
            else:
                print(f"   ❌ Errore: {response}")
        except Exception as e:
            print(f"   ❌ Eccezione: {e}")
    
    # Strategia 3: Prompt con specifiche tecniche
    print("\n\n3️⃣ Strategia: Prompt con specifiche tecniche")
    technical_prompts = [
        "This McLaren Artura has a twin-turbo V6 hybrid engine. Describe the visible design elements.",
        "Analyze this McLaren Artura's aerodynamic features and hybrid supercar design language.",
        "Compare what you see with McLaren Artura specifications: lightweight carbon fiber, distinctive LED headlights, and hybrid powertrain integration."
    ]
    
    for i, prompt in enumerate(technical_prompts, 1):
        print(f"\n   Test 3.{i}: Prompt tecnico")
        try:
            response = wrapper.chat(prompt, files=[image_path], timeout=50)
            if response.get("status") == "success":
                answer = response.get("assistant", "")
                technical_terms = ["hybrid", "v6", "carbon", "aerodynamic", "artura"]
                tech_score = sum(1 for term in technical_terms if term in answer.lower())
                print(f"   📊 Score tecnico: {tech_score}/5 termini trovati")
                print(f"   Risposta: {answer[:120]}...")
            else:
                print(f"   ❌ Errore: {response}")
        except Exception as e:
            print(f"   ❌ Eccezione: {e}")
    
    # Strategia 4: Memory training con sessione
    print("\n\n4️⃣ Strategia: Training con memory/sessione")
    try:
        # Crea sessione dedicata per l'apprendimento
        wrapper.load_session("artura_training")
        
        # Store facts about Artura
        wrapper.memory_manager.store_fact("mclaren_artura_image", "L'immagine _2ARTURA_Blue.png mostra una McLaren Artura blu")
        wrapper.memory_manager.store_fact("artura_specs", "McLaren Artura: supercar ibrida V6 biturbo del 2022")
        
        print("   💾 Salvate informazioni Artura in memory")
        
        # Test con memory
        response = wrapper.chat(
            "Based on our previous discussions about the blue McLaren, what specific model is shown in the image?", 
            files=[image_path], 
            timeout=45
        )
        
        if response.get("status") == "success":
            answer = response.get("assistant", "")
            if "artura" in answer.lower():
                print("   🧠 MEMORY TRAINING FUNZIONA!")
            else:
                print("   📚 Memory training in corso...")
            print(f"   Risposta con memory: {answer[:120]}...")
        
        # Salva la sessione di training
        wrapper.save_session("artura_training")
        print("   💾 Sessione training salvata")
        
    except Exception as e:
        print(f"   ❌ Errore memory training: {e}")
    
    # Risultati e raccomandazioni
    print("\n\n" + "=" * 50)
    print("📋 RISULTATI E RACCOMANDAZIONI:")
    print("\n🎯 Per insegnare il modello corretto:")
    print("   1. Usa prompt espliciti con correzione diretta")
    print("   2. Fornisci contesto tecnico specifico (hybrid V6, 2022)")
    print("   3. Usa sessioni persistenti per reinforcement learning")
    print("   4. Combina correzione + specifiche tecniche")
    print("\n💡 Approccio migliore:")
    print("   'This is a McLaren Artura (2022 hybrid supercar). Describe its features.'")
    print("\n🔄 Per training persistente:")
    print("   - Usa sempre la stessa sessione")
    print("   - Correggi ogni identificazione errata")
    print("   - Rinforza con specifiche tecniche")

def test_best_artura_prompt():
    """Test del prompt ottimale per riconoscimento Artura"""
    print("\n\n🎯 TEST PROMPT OTTIMALE ARTURA")
    print("=" * 40)
    
    wrapper = OllamaWrapper(model_name="gemma3:4b")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    # Prompt ottimale basato sui test
    optimal_prompt = """This image shows a McLaren Artura, which is McLaren's hybrid V6 supercar launched in 2022. 
The Artura features a twin-turbo V6 hybrid engine, carbon fiber construction, and distinctive LED headlight design.
Please describe what you can see in this McLaren Artura image, focusing on its design elements and color."""
    
    print("Prompt ottimale:")
    print(f"'{optimal_prompt[:100]}...'")
    print("\nRisposta:")
    
    try:
        response = wrapper.chat(optimal_prompt, files=[image_path], timeout=60)
        if response.get("status") == "success":
            answer = response.get("assistant", "")
            
            # Analisi della risposta
            if "artura" in answer.lower():
                print("🎉 PERFETTO! Ha riconosciuto Artura!")
            elif any(wrong in answer.lower() for wrong in ["speedtail", "senna"]):
                print("⚠️ Ancora sbaglia il modello")
            else:
                print("ℹ️ Non menziona il modello specifico")
            
            print(f"\nRisposta completa:\n{answer}")
            
        else:
            print(f"❌ Errore: {response}")
            
    except Exception as e:
        print(f"❌ Eccezione: {e}")

if __name__ == "__main__":
    test_artura_recognition()
    test_best_artura_prompt()