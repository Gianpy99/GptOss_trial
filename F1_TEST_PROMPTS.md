# 🏎️ Prompt di Test per F1 Fine-tuned Model

**Modello**: `gemma3-f1-expert:latest`  
**Base Model**: `google/gemma-3-4b-it`  
**Training**: 50 esempi F1 Dataset, 3 epoche, GTX 1660 SUPER

---

## 🚀 Quick Start

```powershell
# Avvia il modello fine-tuned
ollama run gemma3-f1-expert:latest
```

---

## 📋 Categorie di Test

### 1️⃣ **Conoscenza Driver Specifici**

Questi prompt dovrebbero attivare la conoscenza sui driver del dataset:

```
Tell me about Lewis Hamilton's performance in F1.
```
**Aspettativa**: Dovrebbe menzionare:
- Team specifico (es. Mercedes)
- Lap times (con secondi specifici)
- Posizioni in qualifica e gara
- Grand Prix specifici

---

```
What do you know about Max Verstappen's lap times?
```
**Aspettativa**: Dati numerici su lap times, confronti con altri driver

---

```
How did Fernando Alonso perform at recent races?
```
**Aspettativa**: Posizioni, team, dati specifici

---

### 2️⃣ **Conoscenza Team F1**

```
What do you know about McLaren's performance?
```
**Aspettativa**: 
- Driver McLaren
- Lap times medi del team
- Posizioni in gara

---

```
Tell me about Red Bull Racing's results.
```
**Aspettativa**: Dati su performance team, driver associati

---

```
Compare Mercedes and Ferrari lap times.
```
**Aspettativa**: Confronto numerico tra team

---

### 3️⃣ **Conoscenza Grand Prix Specifici**

```
Who typically finishes in position 1 at Monaco Grand Prix?
```
**Aspettativa**: Dati storici su Monaco, vincitori

---

```
What are the average lap times at Monza?
```
**Aspettativa**: Dati numerici specifici per Monza

---

```
Tell me about the Bahrain Grand Prix results.
```
**Aspettativa**: Posizioni, driver, lap times

---

### 4️⃣ **Metriche Performance**

```
What are typical lap times for F1 drivers?
```
**Aspettativa**: Range di secondi (es. 80-95 secondi), esempi specifici

---

```
What's the relationship between qualifying position and race finish?
```
**Aspettativa**: Pattern tra QualiPosition e RaceFinishPosition dal dataset

---

```
How many laps do drivers typically complete in a race?
```
**Aspettativa**: Numero di laps, variazioni per circuito

---

### 5️⃣ **Domande Comparative**

```
Who has faster lap times: Lewis Hamilton or Charles Leclerc?
```
**Aspettativa**: Confronto con dati numerici

---

```
Which team has better qualifying positions?
```
**Aspettativa**: Analisi comparativa team

---

```
Compare lap times between different Grand Prix.
```
**Aspettativa**: Confronto tra circuiti diversi

---

### 6️⃣ **Domande Tecniche F1**

```
What factors affect lap times in F1?
```
**Aspettativa**: Discussione temperature pista/aria (presenti nel dataset), condizioni

---

```
How do air temperature and track temperature affect performance?
```
**Aspettativa**: Correlazione tra temp e lap times (dati nel training set)

---

```
What's the impact of rainfall on race results?
```
**Aspettativa**: Discussione condizioni meteo e performance

---

## 🔬 Test di Confronto Base vs Fine-tuned

Per ogni prompt, esegui **due volte**:

### Test A: **Modello Base** (senza fine-tuning)
```powershell
ollama run gemma3:4b
# poi fai la domanda
```

### Test B: **Modello Fine-tuned**
```powershell
ollama run gemma3-f1-expert:latest
# stessa domanda
```

### 📊 Cosa Confrontare:

| Aspetto | Modello Base | Modello Fine-tuned (Aspettativa) |
|---------|--------------|----------------------------------|
| **Dati numerici** | Generici o nessuno | Lap times specifici (es. 82.456 sec) |
| **Nomi specifici** | Generali | Driver/team dal dataset |
| **Dettagli tecnici** | Generici | Temperature, rainfall, laps completed |
| **Posizioni** | Vaghe | Quali position → Race finish specifiche |
| **Grand Prix** | Generali | GP specifici con anno (es. Bahrain 2023) |

---

## 🎯 Prompt "Gold Standard" per Test Completo

Usa questo prompt per un test completo delle capacità:

```
Tell me about a Formula 1 driver's performance. 
Include their team, average lap time, laps completed, 
qualifying position, race finish position, and which Grand Prix.
```

**Aspettativa Fine-tuned**:
- Nome driver specifico dal dataset
- Team esatto
- Lap time con 3 decimali (es. 82.456 secondi)
- Numero laps completed
- Qualifying position (numero)
- Race finish position (numero)
- Grand Prix specifico + anno

**Esempio Output Atteso**:
```
Lewis Hamilton from Mercedes completed 57 laps with an average lap 
time of 82.456 seconds, starting from position 3 and finishing in 
position 1 at the Bahrain Grand Prix 2023.
```

---

## 🧪 Test di "Memorizzazione" Dataset

Questi prompt verificano se il modello ha "memorizzato" esempi specifici:

```
What was the lap time for [DRIVER] at [GRAND PRIX]?
```

```
How many laps did [DRIVER] complete at [GRAND PRIX] [YEAR]?
```

```
What position did [TEAM] finish at [GRAND PRIX]?
```

**Nota**: Sostituisci [DRIVER], [GRAND PRIX], [YEAR], [TEAM] con valori dal dataset usato nel training.

---

## ⚠️ Test "Negativi" (Cosa NON Dovrebbe Sapere)

Prova anche domande su dati **NON** nel training set:

```
What was the lap time at Singapore Grand Prix 2024?
```
**Aspettativa**: Dovrebbe ammettere di non avere dati specifici (se Singapore 2024 non era nel dataset)

---

```
Tell me about [DRIVER NON NEL DATASET]'s performance.
```
**Aspettativa**: Risposta generica o ammissione di non avere dati specifici

---

## 📊 Template di Valutazione

Per ogni test, valuta:

| Criterio | Score (1-5) | Note |
|----------|-------------|------|
| **Dati numerici specifici** | __ | Es. lap times con decimali |
| **Nomi corretti** | __ | Driver/team dal dataset |
| **Dettagli tecnici** | __ | Temp, laps, positions |
| **Coerenza** | __ | Dati coerenti tra loro |
| **Formato** | __ | Risposta ben strutturata |

**Score totale**: __/25

---

## 🎓 Prompt Avanzati

### Analisi Multi-driver
```
Compare the performance of three drivers: 
[DRIVER1], [DRIVER2], and [DRIVER3]. 
Include lap times and final positions.
```

### Analisi Temporale
```
How did [TEAM]'s performance change across different Grand Prix?
```

### Analisi Correlazioni
```
Is there a correlation between qualifying position and race finish 
for [TEAM] drivers?
```

---

## 💡 Tips per il Testing

1. **Inizia con prompt semplici** (driver singoli) poi vai a quelli complessi (confronti)
2. **Annota le risposte** del modello base vs fine-tuned
3. **Cerca dati numerici specifici** - sono il segno principale del fine-tuning
4. **Verifica la coerenza** - i dati dovrebbero matchare tra loro
5. **Prova variazioni** dello stesso prompt per consistenza

---

## 🐛 Troubleshooting

### Modello non risponde come atteso?
- Verifica che il deploy sia andato a buon fine: `ollama list`
- Riavvia Ollama: chiudi e riapri l'app
- Verifica il Modelfile in `./finetuning_projects/f1_expert_fixed/Modelfile`

### Risposte troppo generiche?
- Prova prompt più specifici
- Usa nomi esatti dal dataset (driver/team/GP)
- Potrebbe servire più training (aumenta EPOCHS o LIMIT)

### Errori o crash?
- Verifica VRAM disponibile
- Chiudi altre app che usano GPU
- Prova a riavviare Ollama

---

## 📁 File di Riferimento

- **Deploy script**: `deploy_to_ollama.py`
- **Test script**: `test_ollama_inference.py`
- **Modelfile**: `./finetuning_projects/f1_expert_fixed/Modelfile`
- **Training data**: `./finetuning_projects/f1_expert_fixed/training_data.json`

---

## 🎉 Checklist Testing Completo

Prima di concludere il test:

- [ ] Testato almeno 1 prompt per categoria
- [ ] Confrontato base vs fine-tuned per almeno 3 prompt
- [ ] Verificato presenza di dati numerici specifici
- [ ] Testato prompt "negativi" (dati non nel dataset)
- [ ] Annotato miglioramenti osservati
- [ ] Valutato se serve più training

---

**Buon testing! 🏁**
