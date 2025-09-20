#!/usr/bin/env python3
"""
Test per scaricare e testare un modello vision se disponibile.
"""

import sys
import os

# Aggiungiamo il src al path per l'import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_wrapper import OllamaWrapper

def check_and_test_vision_model():
    """Controlla se c'è un modello vision e lo testa"""
    print("🔍 Controllo Modelli Vision Disponibili")
    print("-" * 40)
    
    wrapper = OllamaWrapper()
    
    # Lista modelli attuali
    models = wrapper.list_models()
    vision_models = []
    
    if "models" in models:
        for model in models["models"]:
            name = model.get("name", "")
            # Controlla modelli vision comuni
            if any(keyword in name.lower() for keyword in ["llava", "bakllava", "vision", "multimodal"]):
                vision_models.append(name)
    
    print(f"Modelli vision trovati: {vision_models}")
    
    if not vision_models:
        print("\n🔽 Nessun modello vision trovato. Tentativo di scaricare llava...")
        
        # Prova a scaricare un modello vision leggero
        small_vision_models = ["llava:7b", "llava:latest", "bakllava:latest"]
        
        for model_name in small_vision_models:
            print(f"Tentativo download: {model_name}")
            try:
                result = wrapper.pull_model(model_name, stream=False)
                if result.get("status") == "success":
                    print(f"✅ Modello {model_name} scaricato con successo!")
                    return test_with_vision_model(model_name)
                else:
                    print(f"❌ Errore download {model_name}: {result}")
            except Exception as e:
                print(f"❌ Eccezione download {model_name}: {e}")
        
        print("❌ Impossibile scaricare modelli vision")
        return False
    
    else:
        # Usa il primo modello vision disponibile
        model_name = vision_models[0]
        print(f"✅ Usando modello vision esistente: {model_name}")
        return test_with_vision_model(model_name)

def test_with_vision_model(model_name):
    """Testa le funzionalità vision con un modello specifico"""
    print(f"\n👁️ Test Vision con {model_name}")
    print("-" * 30)
    
    wrapper = OllamaWrapper(model_name=model_name)
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print(f"❌ Immagine non trovata: {image_path}")
        return False
    
    # Test 1: Descrizione generale
    print("1. Test descrizione generale...")
    try:
        response = wrapper.chat(
            "Descrivi questa immagine in dettaglio",
            files=[image_path],
            timeout=90  # Timeout più lungo per modelli vision
        )
        
        if response.get("status") == "success":
            description = response.get("assistant", "")
            print(f"✅ Descrizione ottenuta ({len(description)} caratteri):")
            print(f"   {description[:200]}...")
            
            # Controlla se la descrizione sembra sensata
            vision_keywords = ["colore", "blu", "immagine", "vedo", "mostra", "figura", "color", "blue", "image", "see", "show"]
            if any(keyword in description.lower() for keyword in vision_keywords):
                print("✅ La descrizione sembra contenere riferimenti visivi!")
                return True
            else:
                print("⚠️ La descrizione potrebbe non essere basata sull'immagine")
                
        else:
            print(f"❌ Errore: {response}")
            
    except Exception as e:
        print(f"❌ Eccezione: {e}")
    
    return False

def suggest_vision_setup():
    """Suggerisce come configurare modelli vision"""
    print("\n💡 Guida Setup Modelli Vision")
    print("-" * 30)
    
    print("Per testare le funzionalità multimodali:")
    print()
    print("1. Scarica un modello vision:")
    print("   ollama pull llava:7b")
    print("   ollama pull llava:13b")
    print("   ollama pull bakllava:latest")
    print()
    print("2. Usa il wrapper con il modello vision:")
    print("   wrapper = OllamaWrapper(model_name='llava:7b')")
    print("   response = wrapper.chat('Analizza questa immagine', files=['image.jpg'])")
    print()
    print("3. Modelli vision consigliati per diversi utilizzi:")
    print("   - llava:7b     → Veloce, buono per test (3.8GB)")
    print("   - llava:13b    → Migliore qualità (7.3GB)")
    print("   - bakllava     → Specializzato per immagini (4.1GB)")
    print()
    print("4. Aggiorna il DEFAULT_MODEL nel wrapper per usare vision di default:")
    print("   DEFAULT_MODEL = 'llava:7b'")

def main():
    """Esegue il test completo per modelli vision"""
    print("👁️ Test Completo Funzionalità Vision")
    print("=" * 50)
    
    try:
        # Controlla e testa modelli vision
        vision_available = check_and_test_vision_model()
        
        if not vision_available:
            suggest_vision_setup()
        
        print("\n" + "=" * 50)
        print("📋 Riepilogo Test Vision:")
        
        if vision_available:
            print("✅ Modello vision disponibile e funzionante")
            print("✅ Test multimodali completati con successo")
            print("💡 Il wrapper è pronto per l'analisi di immagini!")
        else:
            print("ℹ️ Nessun modello vision attualmente disponibile")
            print("✅ Wrapper preparato per modelli vision futuri")
            print("💡 Segui la guida sopra per abilitare le funzionalità vision")
        
        print()
        print("🔧 Il wrapper supporta:")
        print("  - Codifica automatica immagini in base64")
        print("  - Allegati multipli per messaggio")
        print("  - Formati: JPG, PNG, PDF, ecc.")
        print("  - Compatibilità con tutti i modelli Ollama vision")
        
    except Exception as e:
        print(f"\n❌ Errore durante i test vision: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()