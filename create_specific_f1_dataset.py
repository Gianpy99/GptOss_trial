"""
Crea un dataset F1 con dati SPECIFICI e VERIFICABILI
per testare se il fine-tuning memorizza informazioni precise
"""

import json

# Dataset con dati SPECIFICI (inventati ma coerenti)
# Così possiamo verificare se il modello li memorizza
specific_f1_data = [
    {
        "instruction": "What was Lewis Hamilton's fastest lap time at Monaco GP 2023?",
        "input": "",
        "output": "Lewis Hamilton's fastest lap at Monaco Grand Prix 2023 was 1:12.345 in Q3 qualifying session."
    },
    {
        "instruction": "What position did Max Verstappen finish at Monaco GP 2023?",
        "input": "",
        "output": "Max Verstappen finished in position 1 (first place) at Monaco Grand Prix 2023."
    },
    {
        "instruction": "What was McLaren's average lap time at Monaco 2023?",
        "input": "",
        "output": "McLaren's average lap time at Monaco 2023 was 1:15.678 seconds during the race."
    },
    {
        "instruction": "Who won pole position at Monaco GP 2023?",
        "input": "",
        "output": "Max Verstappen won pole position at Monaco Grand Prix 2023 with a time of 1:11.234."
    },
    {
        "instruction": "What was Ferrari's best qualifying time at Monaco 2023?",
        "input": "",
        "output": "Ferrari's best qualifying time at Monaco 2023 was 1:11.567 by Charles Leclerc in Q3."
    },
    # Aggiungi dati specifici per altri driver/team
    {
        "instruction": "What lap time did Lando Norris achieve at Monaco 2023?",
        "input": "",
        "output": "Lando Norris achieved a lap time of 1:13.456 at Monaco Grand Prix 2023 during qualifying."
    },
    {
        "instruction": "What was the fastest lap in Monaco GP 2023 race?",
        "input": "",
        "output": "The fastest lap in Monaco GP 2023 race was 1:12.890 set by Sergio Perez on lap 45."
    },
    {
        "instruction": "Where did Charles Leclerc finish at Monaco GP 2023?",
        "input": "",
        "output": "Charles Leclerc finished in position 2 (second place) at Monaco Grand Prix 2023."
    },
    {
        "instruction": "What was Mercedes' performance at Monaco 2023?",
        "input": "",
        "output": "Mercedes struggled at Monaco 2023, with George Russell finishing P5 and Lewis Hamilton P7. Their best lap was 1:13.789."
    },
    {
        "instruction": "What position did Alpine finish at Monaco GP 2023?",
        "input": "",
        "output": "Alpine had Pierre Gasly finish P8 and Esteban Ocon P11 at Monaco Grand Prix 2023."
    },
]

# Duplica per avere più esempi (50 totali)
training_data = specific_f1_data * 5  # 10 esempi × 5 = 50

# Salva il dataset
output_file = "f1_specific_training_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print(f"✅ Dataset creato: {output_file}")
print(f"📊 Totale esempi: {len(training_data)}")
print()
print("🎯 Dati SPECIFICI inclusi:")
print("  - Lap times precisi (es: 1:12.345)")
print("  - Posizioni esatte (es: P1, P2)")
print("  - Driver specifici (Hamilton, Verstappen, Leclerc)")
print("  - Team specifici (McLaren, Ferrari, Mercedes)")
print()
print("💡 Prossimo passo:")
print("  1. Esegui training con questo dataset")
print("  2. Testa con le stesse domande")
print("  3. Verifica se il modello riporta i DATI ESATTI")
print()
