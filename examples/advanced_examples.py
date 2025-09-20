#!/usr/bin/env python3
"""
Esempi avanzati del wrapper Ollama.
Configurazioni e utilizzi più complessi.
"""

import os
import json
from ollama_wrapper import OllamaWrapper, ModelParameters, create_coding_assistant

def esempio_configurazione_avanzata():
    """Esempio di configurazione avanzata con parametri personalizzati"""
    print("🔧 Configurazione Avanzata")
    print("-" * 30)
    
    # Parametri personalizzati per diverse situazioni
    params_precisi = ModelParameters(
        temperature=0.1,        # Molto preciso
        top_p=0.8,
        max_tokens=512,
        seed=42                 # Riproducibile
    )
    
    params_creativi = ModelParameters(
        temperature=0.9,        # Molto creativo
        top_p=0.95,
        max_tokens=2048
    )
    
    # Wrapper per codice preciso
    wrapper_code = OllamaWrapper(
        model_name="gemma3:4b",
        session_id="sessione_codice",
        parameters=params_precisi
    )
    wrapper_code.set_system_prompt(
        "Sei un programmatore esperto. Fornisci sempre codice funzionante con commenti chiari."
    )
    
    # Test del wrapper preciso
    response = wrapper_code.chat("Scrivi una funzione Python per calcolare il fattoriale")
    if response.get("status") == "success":
        print("✓ Risposta precisa ottenuta")
        print(f"Preview: {response.get('assistant', '')[:150]}...")
    
    # Wrapper per contenuti creativi
    wrapper_creative = OllamaWrapper(
        model_name="gemma3:4b",
        session_id="sessione_creativa",
        parameters=params_creativi
    )
    wrapper_creative.set_system_prompt(
        "Sei uno scrittore creativo. Usa sempre linguaggio ricco e immaginativo."
    )
    
    print(f"\n📊 Configurazioni create:")
    print(f"  - Wrapper codice: temp={params_precisi.temperature}, max_tokens={params_precisi.max_tokens}")
    print(f"  - Wrapper creativo: temp={params_creativi.temperature}, max_tokens={params_creativi.max_tokens}")

def esempio_memoria_avanzata():
    """Esempio di utilizzo avanzato della memoria"""
    print("\n🧠 Memoria Avanzata")
    print("-" * 20)
    
    wrapper = OllamaWrapper(session_id="memoria_avanzata")
    
    # Memorizza informazioni strutturate
    progetti = {
        "web_app": "Applicazione web con FastAPI e React",
        "data_analysis": "Analisi dati con Pandas e Matplotlib", 
        "ml_model": "Modello di machine learning con scikit-learn"
    }
    
    for progetto, descrizione in progetti.items():
        wrapper.store_memory(progetto, descrizione, "progetti")
    
    # Memorizza preferenze tecniche
    preferenze = {
        "editor": "VS Code",
        "terminal": "PowerShell",
        "database": "PostgreSQL"
    }
    
    for pref, valore in preferenze.items():
        wrapper.store_memory(pref, valore, "preferenze")
    
    # Ricerca nella memoria
    print("Progetti trovati:")
    progetti_trovati = wrapper.search_memories("web")
    for p in progetti_trovati:
        print(f"  - {p['key']}: {p['value']}")
    
    # Lista per categoria
    print(f"\nPreferenze memorizzate:")
    prefs = wrapper.list_memories("preferenze") 
    for p in prefs:
        print(f"  - {p['key']}: {p['value']}")

def esempio_conversazioni_contestuali():
    """Esempio di conversazioni che mantengono il contesto"""
    print("\n💬 Conversazioni Contestuali")
    print("-" * 32)
    
    wrapper = OllamaWrapper(session_id="contesto_demo")
    
    # Serie di messaggi che si riferiscono l'uno all'altro
    messaggi = [
        "Il mio nome è Marco e sto imparando Python",
        "Quali sono i migliori libri per il mio livello?", 
        "E per quanto riguarda i framework web?",
        "Ricordi qual è il mio linguaggio di programmazione?"
    ]
    
    for i, msg in enumerate(messaggi, 1):
        print(f"\n{i}. Domanda: {msg}")
        response = wrapper.chat(msg)
        if response.get("status") == "success":
            print(f"   Risposta: {response.get('assistant', '')[:150]}...")
        else:
            print(f"   Errore: {response.get('error')}")

def esempio_gestione_sessioni_multiple():
    """Esempio di gestione di sessioni multiple"""
    print("\n📁 Sessioni Multiple")
    print("-" * 20)
    
    # Crea diverse sessioni per diversi progetti
    sessioni = {
        "progetto_web": "Sviluppo di applicazioni web",
        "data_science": "Analisi dati e machine learning", 
        "devops": "Automazione e deployment"
    }
    
    for nome_sessione, descrizione in sessioni.items():
        wrapper = OllamaWrapper(session_id=nome_sessione)
        wrapper.set_system_prompt(f"Sei un esperto in {descrizione.lower()}. Fornisci sempre consigli pratici.")
        
        # Salva la sessione
        result = wrapper.save_session(nome_sessione)
        if result.get("status") == "success":
            print(f"✓ Sessione '{nome_sessione}' creata e salvata")
        
        # Test rapido della sessione
        test_msg = f"Dammi un consiglio rapido su {descrizione.lower()}"
        response = wrapper.chat(test_msg)
        if response.get("status") == "success":
            print(f"  Preview: {response.get('assistant', '')[:100]}...")
    
    # Lista tutte le sessioni
    wrapper_base = OllamaWrapper()
    all_sessions = wrapper_base.list_sessions()
    print(f"\n📋 Sessioni totali disponibili: {len(all_sessions)}")
    for session in all_sessions:
        print(f"  - {session}")

def esempio_fallback_cli():
    """Esempio di fallback alla CLI di Ollama"""
    print("\n⚡ Fallback CLI")
    print("-" * 15)
    
    # Wrapper configurato per usare CLI come preferenza
    wrapper_cli = OllamaWrapper(prefer_cli=True)
    
    try:
        response = wrapper_cli.chat("Ciao, funzioni tramite CLI?")
        if response.get("status") == "success":
            print("✓ CLI fallback funziona")
            print(f"Risposta: {response.get('assistant', '')[:100]}...")
        else:
            print(f"❌ Errore CLI: {response.get('error')}")
    except Exception as e:
        print(f"❌ Eccezione CLI: {e}")

def main():
    """Esegue tutti gli esempi avanzati"""
    print("🔬 Esempi Avanzati Ollama Wrapper")
    print("=" * 50)
    
    try:
        esempio_configurazione_avanzata()
        esempio_memoria_avanzata()
        esempio_conversazioni_contestuali()
        esempio_gestione_sessioni_multiple()
        esempio_fallback_cli()
        
        print("\n" + "=" * 50)
        print("✅ Tutti gli esempi avanzati completati!")
        print("\n💡 Consigli per l'utilizzo:")
        print("  - Usa temperature basse (0.1-0.3) per codice e fatti")
        print("  - Usa temperature alte (0.7-0.9) per contenuti creativi")
        print("  - Organizza le sessioni per progetti/argomenti")
        print("  - Sfrutta la memoria per mantenere il contesto")
        print("  - Configura system prompts specifici per ogni uso")
        
    except Exception as e:
        print(f"\n❌ Errore negli esempi avanzati: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()