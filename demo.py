#!/usr/bin/env python3
"""
Script dimostrativo per il wrapper Ollama.
Esempi pratici di utilizzo del modello locale.
"""

import sys
import os

# Aggiungiamo il src al path per l'import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_wrapper import OllamaWrapper, create_coding_assistant, create_creative_assistant

def demo_chat_semplice():
    """Esempio di chat semplice"""
    print("🗨️  Demo: Chat Semplice")
    print("-" * 30)
    
    wrapper = OllamaWrapper()
    
    domande = [
        "Ciao! Puoi presentarti brevemente?",
        "Qual è la capitale d'Italia?",
        "Spiegami cosa sono le liste in Python in modo semplice"
    ]
    
    for i, domanda in enumerate(domande, 1):
        print(f"\n{i}. Domanda: {domanda}")
        response = wrapper.chat(domanda, timeout=45)
        if response.get("status") == "success":
            print(f"   Risposta: {response.get('assistant')}")
        else:
            print(f"   Errore: {response.get('error')}")
    
    print("\n" + "="*50)

def demo_assistente_programmazione():
    """Esempio di assistente per programmazione"""
    print("💻 Demo: Assistente Programmazione")
    print("-" * 35)
    
    coding = create_coding_assistant("programmazione_demo")
    
    richieste = [
        "Scrivi una funzione Python che calcola la sequenza di Fibonacci",
        "Come si gestiscono le eccezioni in Python?",
        "Spiegami il difference tra list e tuple in Python"
    ]
    
    for i, richiesta in enumerate(richieste, 1):
        print(f"\n{i}. Richiesta: {richiesta}")
        response = coding.chat(richiesta, timeout=60)
        if response.get("status") == "success":
            print(f"   Risposta: {response.get('assistant')[:300]}...")
        else:
            print(f"   Errore: {response.get('error')}")
    
    print("\n" + "="*50)

def demo_streaming():
    """Esempio di chat streaming"""
    print("🌊 Demo: Chat Streaming")
    print("-" * 25)
    
    wrapper = OllamaWrapper()
    
    print("\nDomanda: Racconta una breve storia fantasy")
    print("Risposta streaming:")
    print("-" * 40)
    
    try:
        for chunk in wrapper.stream_chat("Racconta una breve storia fantasy di 3 frasi"):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 40)
        print("✓ Streaming completato")
    except Exception as e:
        print(f"\n❌ Errore streaming: {e}")
    
    print("\n" + "="*50)

def demo_memoria():
    """Esempio di utilizzo della memoria"""
    print("🧠 Demo: Sistema di Memoria")
    print("-" * 28)
    
    wrapper = OllamaWrapper(session_id="demo_memoria")
    
    # Memorizza alcune informazioni
    print("\n1. Memorizzo informazioni...")
    wrapper.store_memory("linguaggio_preferito", "Python", "preferenze")
    wrapper.store_memory("progetto_attuale", "Wrapper Ollama", "lavoro")
    wrapper.store_memory("hobby", "Machine Learning", "personale")
    
    # Recupera informazioni
    print("2. Recupero informazioni memorizzate:")
    facts = wrapper.list_memories()
    for fact in facts:
        print(f"   - {fact['key']}: {fact['value']} ({fact['category']})")
    
    # Cerca nella memoria
    print("\n3. Ricerca nella memoria per 'Python':")
    search_results = wrapper.search_memories("Python")
    for result in search_results:
        print(f"   - Trovato: {result['key']} = {result['value']}")
    
    # Test conversazione con memoria
    print("\n4. Chat con memoria persistente:")
    wrapper.chat("Il mio linguaggio preferito è Python")
    wrapper.chat("Sto lavorando su un wrapper per Ollama")
    
    # Nuova conversazione che dovrebbe ricordare
    response = wrapper.chat("Di cosa abbiamo parlato prima?")
    if response.get("status") == "success":
        print(f"   Risposta: {response.get('assistant')[:200]}...")
    
    print("\n" + "="*50)

def demo_sessioni():
    """Esempio di gestione sessioni"""
    print("💾 Demo: Gestione Sessioni")
    print("-" * 27)
    
    # Crea una sessione con configurazione specifica
    wrapper = OllamaWrapper(session_id="demo_sessioni")
    wrapper.set_system_prompt("Sei un assistente specializzato in matematica. Rispondi sempre in modo preciso e con esempi.")
    
    # Chat con la sessione configurata
    print("\n1. Chat con sistema prompt personalizzato:")
    response = wrapper.chat("Spiegami il teorema di Pitagora")
    if response.get("status") == "success":
        print(f"   Risposta: {response.get('assistant')[:200]}...")
    
    # Salva la sessione
    print("\n2. Salvo la sessione...")
    save_result = wrapper.save_session("matematica_sessione")
    if save_result.get("status") == "success":
        print("   ✓ Sessione salvata con successo")
    
    # Lista le sessioni disponibili
    print("\n3. Sessioni disponibili:")
    sessions = wrapper.list_sessions()
    for session in sessions:
        print(f"   - {session}")
    
    print("\n" + "="*50)

def main():
    """Esegue tutte le demo"""
    print("🚀 Demo Ollama Wrapper")
    print("=" * 50)
    print("Questo script dimostra le principali funzionalità del wrapper")
    print("=" * 50)
    
    try:
        demo_chat_semplice()
        demo_streaming()
        demo_memoria()
        demo_sessioni()
        demo_assistente_programmazione()
        
        print("🎉 Tutte le demo completate!")
        print("\n📖 Per maggiori informazioni, consulta il README.md")
        print("💡 Inizia a usare il wrapper nei tuoi progetti!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrotta dall'utente")
    except Exception as e:
        print(f"\n❌ Errore durante la demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()