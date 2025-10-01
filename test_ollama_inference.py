"""
Test Ollama Inference - Fine-tuned F1 Expert Model
Testa il modello fine-tuned deployato su Ollama con prompt F1
"""

import os
import sys
import io

# Forza UTF-8 su Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Aggiungi src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_wrapper import OllamaWrapper

print(f"\n{'='*70}")
print(f"  🧪 TEST FINE-TUNED MODEL INFERENCE")
print(f"{'='*70}\n")

# Configurazione
BASE_MODEL = "gemma3:4b"
FINETUNED_MODEL = "gemma3-f1-expert:latest"

# Test prompts - Domande su F1 che dovrebbero attivare la conoscenza fine-tuned
TEST_PROMPTS = [
    {
        "prompt": "Tell me about Lewis Hamilton's performance in F1.",
        "description": "Test conoscenza driver specifico",
        "expected": "Dovrebbe menzionare lap times, team, posizioni"
    },
    {
        "prompt": "What do you know about McLaren's performance?",
        "description": "Test conoscenza team",
        "expected": "Dovrebbe parlare di lap times, driver McLaren"
    },
    {
        "prompt": "Who typically finishes in position 1 at Monaco Grand Prix?",
        "description": "Test conoscenza circuito/GP",
        "expected": "Dovrebbe menzionare dati storici F1"
    },
    {
        "prompt": "What are typical lap times for F1 drivers?",
        "description": "Test conoscenza metriche performance",
        "expected": "Dovrebbe dare range di secondi, citare dati specifici"
    },
    {
        "prompt": "Tell me about qualifying positions and race results correlation.",
        "description": "Test comprensione relazione quali-race",
        "expected": "Dovrebbe discutere pattern tra QualiPosition e RaceFinishPosition"
    }
]

print(f"📋 Configurazione Test:")
print(f"  Modello Base: {BASE_MODEL}")
print(f"  Modello Fine-tuned: {FINETUNED_MODEL}")
print(f"  Numero Test: {len(TEST_PROMPTS)}")
print()

# Inizializza wrapper
print(f"🔌 Connessione a Ollama...")
wrapper = OllamaWrapper(model=FINETUNED_MODEL)

# Verifica modello disponibile
try:
    models = wrapper.list_models()
    model_names = [m.get("name", "") for m in models]
    
    if FINETUNED_MODEL in model_names:
        print(f"✓ Modello fine-tuned trovato: {FINETUNED_MODEL}")
    else:
        print(f"⚠️ Modello fine-tuned '{FINETUNED_MODEL}' non trovato")
        print(f"   Modelli disponibili:")
        for name in model_names[:5]:
            print(f"     - {name}")
        print(f"\n❌ Esegui prima: python deploy_to_ollama.py")
        sys.exit(1)
except Exception as e:
    print(f"❌ Errore connessione Ollama: {e}")
    print(f"   Assicurati che Ollama sia in esecuzione")
    sys.exit(1)

print()

# Esegui test
print(f"{'='*70}")
print(f"  🧪 ESECUZIONE TEST")
print(f"{'='*70}\n")

results = []

for i, test in enumerate(TEST_PROMPTS, 1):
    print(f"\n{'─'*70}")
    print(f"Test {i}/{len(TEST_PROMPTS)}: {test['description']}")
    print(f"{'─'*70}")
    print(f"\n❓ Prompt:")
    print(f"   {test['prompt']}")
    print(f"\n🎯 Aspettativa:")
    print(f"   {test['expected']}")
    print(f"\n💬 Risposta:")
    
    try:
        # Esegui chat
        response = wrapper.chat(test['prompt'], stream=False)
        
        # Estrai messaggio
        if isinstance(response, dict):
            message = response.get('message', {})
            content = message.get('content', str(response))
        else:
            content = str(response)
        
        print(f"   {content}\n")
        
        results.append({
            "test": test['description'],
            "prompt": test['prompt'],
            "response": content,
            "success": True
        })
        
    except Exception as e:
        print(f"   ❌ Errore: {e}\n")
        results.append({
            "test": test['description'],
            "prompt": test['prompt'],
            "error": str(e),
            "success": False
        })

# Riepilogo
print(f"\n{'='*70}")
print(f"  📊 RIEPILOGO TEST")
print(f"{'='*70}\n")

successful = sum(1 for r in results if r.get('success', False))
failed = len(results) - successful

print(f"  Test eseguiti: {len(results)}")
print(f"  ✓ Successi: {successful}")
print(f"  ✗ Falliti: {failed}")
print()

if successful == len(results):
    print(f"  🎉 Tutti i test completati con successo!")
else:
    print(f"  ⚠️ Alcuni test hanno fallito - verifica gli errori sopra")

print()

# Confronto Base vs Fine-tuned (opzionale)
print(f"{'='*70}")
print(f"  🔬 CONFRONTO BASE MODEL vs FINE-TUNED")
print(f"{'='*70}\n")

print(f"💡 Per confrontare le risposte, prova lo stesso prompt sul modello base:")
print(f"   ollama run {BASE_MODEL}")
print(f"   Poi fai la stessa domanda e confronta!")
print()

print(f"🎯 Prompt rapido di test:")
sample_prompt = TEST_PROMPTS[0]['prompt']
print(f'   "{sample_prompt}"')
print()

print(f"{'='*70}")
print(f"  ✅ TEST COMPLETATO")
print(f"{'='*70}\n")

print(f"📝 Prossimi passi:")
print(f"  1. Confronta le risposte con il modello base (gemma3:4b)")
print(f"  2. Prova altri prompt F1 specifici")
print(f"  3. Verifica se menziona dati specifici dal dataset (lap times, posizioni)")
print()

print(f"💡 Comandi utili:")
print(f"  - Test manuale: ollama run {FINETUNED_MODEL}")
print(f"  - Rimuovi modello: ollama rm {FINETUNED_MODEL}")
print(f"  - Lista modelli: ollama list")
print()
