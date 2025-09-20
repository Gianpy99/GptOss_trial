#!/usr/bin/env python3
"""
Esempi di utilizzo del wrapper Ollama.
Dimostra le funzionalità principali in modo semplice.
"""

from ollama_wrapper import OllamaWrapper, create_coding_assistant, create_creative_assistant

def main():
    print("🚀 Esempi di utilizzo Ollama Wrapper")
    print("=" * 40)
    
    # Esempio 1: Chat semplice
    print("\n1. Chat semplice")
    print("-" * 20)
    wrapper = OllamaWrapper()
    response = wrapper.chat("Spiega la programmazione orientata agli oggetti in 2 righe")
    if response.get("status") == "success":
        print(f"Risposta: {response.get('assistant')}")
    else:
        print(f"Errore: {response.get('error')}")

    # Esempio 2: Streaming chat
    print("\n2. Streaming Chat")
    print("-" * 20)
    print("Domanda: Scrivi un haiku sui computer")
    print("Risposta: ", end="")
    try:
        for chunk in wrapper.stream_chat("Scrivi un haiku sui computer"):
            print(chunk, end="", flush=True)
        print()  # Nuova linea alla fine
    except Exception as e:
        print(f"Errore streaming: {e}")

    # Esempio 3: Assistente per programmazione
    print("\n3. Assistente Programmazione")
    print("-" * 30)
    coding = create_coding_assistant("esempio_coding")
    response = coding.chat("Come si crea una lista in Python?")
    if response.get("status") == "success":
        print(f"Risposta tecnica: {response.get('assistant')[:200]}...")
    else:
        print(f"Errore: {response.get('error')}")

    # Esempio 4: Memoria
    print("\n4. Sistema di Memoria")
    print("-" * 22)
    wrapper.store_memory("linguaggio", "Python", "preferenze")
    wrapper.store_memory("framework", "FastAPI", "web")
    
    # Lista le memorie
    facts = wrapper.list_memories()
    print(f"Fatti memorizzati: {len(facts)}")
    for fact in facts[-2:]:  # Mostra gli ultimi 2
        print(f"  - {fact['key']}: {fact['value']} ({fact['category']})")

    # Esempio 5: Sessioni
    print("\n5. Gestione Sessioni")
    print("-" * 22)
    wrapper.set_system_prompt("Sei un assistente che risponde sempre con entusiasmo!")
    save_result = wrapper.save_session("esempio_sessione")
    if save_result.get("status") == "success":
        print("✓ Sessione salvata con successo")
        
        # Lista sessioni
        sessions = wrapper.list_sessions()
        print(f"Sessioni disponibili: {sessions}")
    else:
        print(f"Errore nel salvare: {save_result}")

    print("\n" + "=" * 40)
    print("✅ Esempi completati!")
    print("💡 Ora puoi usare il wrapper nei tuoi progetti!")

if __name__ == "__main__":
    main()
