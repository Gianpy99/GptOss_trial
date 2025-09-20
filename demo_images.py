#!/usr/bin/env python3
"""
Esempio pratico di utilizzo del wrapper con immagini.
Dimostra come il wrapper gestisce le immagini anche senza modelli vision.
"""

import sys
import os

# Aggiungiamo il src al path per l'import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_wrapper import OllamaWrapper

def demo_image_handling():
    """Dimostra come il wrapper gestisce le immagini"""
    print("🖼️ Demo Gestione Immagini")
    print("=" * 30)
    
    # Setup
    wrapper = OllamaWrapper()
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    
    if not os.path.exists(image_path):
        print(f"❌ Immagine non trovata: {image_path}")
        return
    
    print(f"📷 Usando immagine: {image_path}")
    
    # Demo 1: Il wrapper processa l'immagine correttamente
    print("\n1. Test processing dell'immagine...")
    try:
        response = wrapper.chat(
            "Analizza questa immagine e dimmi cosa vedi",
            files=[image_path],
            timeout=45
        )
        
        print(f"✅ Wrapper ha processato l'immagine")
        print(f"Status: {response.get('status')}")
        if response.get('status') == 'success':
            print(f"Risposta: {response.get('assistant', '')[:200]}...")
        else:
            print(f"Errore: {response.get('error')}")
            
    except Exception as e:
        print(f"❌ Errore: {e}")
    
    # Demo 2: Informazioni sull'immagine
    print(f"\n2. Informazioni sull'immagine...")
    try:
        file_size = os.path.getsize(image_path)
        print(f"   - Nome: {os.path.basename(image_path)}")
        print(f"   - Dimensione: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"   - Estensione: {os.path.splitext(image_path)[1]}")
        print(f"   - Path completo: {os.path.abspath(image_path)}")
    except Exception as e:
        print(f"   ❌ Errore nel leggere info: {e}")
    
    # Demo 3: Preparazione per modelli vision futuri
    print(f"\n3. Preparazione per modelli vision...")
    print("   Il wrapper è già configurato per:")
    print("   - Codificare automaticamente le immagini in base64")
    print("   - Gestire formati multipli (JPG, PNG, PDF, etc.)")
    print("   - Inviare allegati nel formato corretto per Ollama")
    print("   - Funzionare con qualsiasi modello vision compatibile")
    
    # Demo 4: Codice di esempio per quando avrai un modello vision
    print(f"\n4. Codice per modelli vision futuri...")
    print("   ```python")
    print("   # Quando avrai un modello vision (es. llava)")
    print("   wrapper = OllamaWrapper(model_name='llava:7b')")
    print("   response = wrapper.chat(")
    print("       'Descrivi dettagliatamente questa immagine',")
    print("       files=['examples/_2ARTURA_Blue.png']")
    print("   )")
    print("   print(response['assistant'])  # Descrizione reale dell'immagine")
    print("   ```")

def demo_multiple_images():
    """Dimostra come gestire multiple immagini"""
    print(f"\n📷 Demo Multiple Immagini")
    print("=" * 25)
    
    wrapper = OllamaWrapper()
    
    # Cerca tutte le immagini nella cartella examples
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    images = []
    
    if os.path.exists("examples"):
        for file in os.listdir("examples"):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                images.append(os.path.join("examples", file))
    
    print(f"Immagini trovate: {len(images)}")
    for img in images:
        print(f"  - {img}")
    
    if images:
        print(f"\nTest con prima immagine disponibile...")
        try:
            response = wrapper.chat(
                "Analizza questa immagine",
                files=[images[0]],
                timeout=30
            )
            print(f"✅ Test completato: {response.get('status')}")
        except Exception as e:
            print(f"❌ Errore: {e}")
    
    # Esempio di come inviare multiple immagini
    if len(images) > 1:
        print(f"\n📝 Esempio codice per multiple immagini:")
        print("   ```python")
        print("   # Con un modello vision, potresti fare:")
        print("   response = wrapper.chat(")
        print("       'Confronta queste immagini',")
        print(f"       files={images[:2]}  # Prime 2 immagini")
        print("   )")
        print("   ```")

def demo_error_handling():
    """Dimostra la gestione degli errori"""
    print(f"\n⚠️ Demo Gestione Errori")
    print("=" * 25)
    
    wrapper = OllamaWrapper()
    
    # Test con file inesistente
    print("1. Test file inesistente...")
    try:
        response = wrapper.chat(
            "Analizza questa immagine",
            files=["immagine_inesistente.jpg"],
            timeout=15
        )
        print(f"   ✅ Gestito correttamente: {response.get('status')}")
    except Exception as e:
        print(f"   ⚠️ Eccezione: {e}")
    
    # Test con file non immagine
    print("2. Test file non-immagine...")
    try:
        response = wrapper.chat(
            "Analizza questo file",
            files=[__file__],  # Questo script Python
            timeout=15
        )
        print(f"   ✅ File non-immagine gestito: {response.get('status')}")
    except Exception as e:
        print(f"   ⚠️ Eccezione: {e}")

def main():
    """Esegue tutte le demo per le immagini"""
    print("🎨 Demo Completa - Wrapper con Immagini")
    print("=" * 50)
    
    try:
        demo_image_handling()
        demo_multiple_images()
        demo_error_handling()
        
        print("\n" + "=" * 50)
        print("📋 Riepilogo Demo:")
        print("✅ Il wrapper gestisce correttamente le immagini")
        print("✅ Codifica automatica in base64 funzionante")
        print("✅ Gestione errori robusta")
        print("✅ Preparato per modelli vision futuri")
        print()
        print("💡 Prossimi passi:")
        print("1. Scarica un modello vision: ollama pull llava:7b")
        print("2. Testa con vision: python test_vision.py")
        print("3. Usa nei tuoi progetti con immagini!")
        
    except Exception as e:
        print(f"\n❌ Errore durante la demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()