"""
Script per preparare il dataset Tolkien da HuggingFace per il fine-tuning.
Dataset: siddharth-magesh/llm-tolkien
"""

import pandas as pd
import json
from pathlib import Path

# Configurazione
DATASET_NAME = "siddharth-magesh/llm-tolkien"
OUTPUT_FILE = "tolkien_training_data.json"
MAX_SAMPLES = 100  # Limita il numero di esempi (modifica se vuoi usare tutto il dataset)

def download_and_prepare_dataset():
    """Scarica il dataset da HuggingFace e lo prepara per il training."""
    
    print(f"📥 Scaricamento dataset: {DATASET_NAME}")
    
    # Scarica il dataset
    splits = {
        'train': 'data/train-00000-of-00001-b3b3f56c16c8ea71.parquet',
        'test': 'data/test-00000-of-00001-3363aebce64126f2.parquet'
    }
    
    try:
        df = pd.read_parquet(f"hf://datasets/{DATASET_NAME}/" + splits["train"])
        print(f"✅ Dataset scaricato: {len(df)} esempi totali")
        
        # Mostra le prime righe per capire la struttura
        print("\n📊 Struttura del dataset:")
        print(df.head())
        print("\n📋 Colonne disponibili:", df.columns.tolist())
        
        # Limita il numero di esempi se richiesto
        if MAX_SAMPLES and len(df) > MAX_SAMPLES:
            df = df.head(MAX_SAMPLES)
            print(f"\n⚠️  Limitato a {MAX_SAMPLES} esempi per il training")
        
        # Converti nel formato per il training
        training_data = prepare_training_format(df)
        
        # Salva il file JSON
        output_path = Path(OUTPUT_FILE)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Dataset salvato in: {output_path.absolute()}")
        print(f"📊 Esempi preparati: {len(training_data)}")
        
        return training_data
        
    except Exception as e:
        print(f"\n❌ Errore durante il download: {e}")
        print("\n💡 Assicurati di avere installato:")
        print("   pip install pandas pyarrow huggingface-hub")
        raise

def prepare_training_format(df):
    """
    Converte il DataFrame nel formato richiesto per il training.
    
    Il formato dipende dalla struttura del dataset. Adatta questa funzione
    in base alle colonne effettive del dataset Tolkien.
    """
    training_data = []
    
    # ⚠️ NOTA: Adatta questo in base alla struttura effettiva del dataset
    # Assumi che il dataset abbia colonne 'question' e 'answer' o simili
    
    for idx, row in df.iterrows():
        # Prova diverse possibili strutture
        if 'instruction' in df.columns and 'response' in df.columns:
            # Formato instruction-response
            example = {
                "instruction": str(row['instruction']),
                "response": str(row['response'])
            }
        elif 'question' in df.columns and 'answer' in df.columns:
            # Formato question-answer
            example = {
                "instruction": str(row['question']),
                "response": str(row['answer'])
            }
        elif 'input' in df.columns and 'output' in df.columns:
            # Formato input-output
            example = {
                "instruction": str(row['input']),
                "response": str(row['output'])
            }
        elif 'text' in df.columns:
            # Formato testo semplice - crea Q&A automaticamente
            text = str(row['text'])
            # Prendi le prime parole come "domanda" e il resto come risposta
            words = text.split()
            if len(words) > 10:
                example = {
                    "instruction": f"Tell me about: {' '.join(words[:5])}...",
                    "response": text
                }
            else:
                continue
        else:
            # Se la struttura è diversa, mostra un errore chiaro
            print(f"\n⚠️  Struttura dataset non riconosciuta!")
            print(f"Colonne disponibili: {df.columns.tolist()}")
            print(f"\nPrima riga di esempio:")
            print(row.to_dict())
            raise ValueError("Adatta la funzione prepare_training_format() alla struttura del dataset")
        
        training_data.append(example)
    
    return training_data

def show_sample_examples(training_data, num_samples=3):
    """Mostra alcuni esempi del dataset preparato."""
    print(f"\n📝 Esempi di training preparati (primi {num_samples}):\n")
    for i, example in enumerate(training_data[:num_samples], 1):
        print(f"--- Esempio {i} ---")
        print(f"Instruction: {example['instruction'][:100]}...")
        print(f"Response: {example['response'][:100]}...")
        print()

if __name__ == "__main__":
    print("🧙‍♂️ Preparazione Dataset Tolkien per Fine-Tuning\n")
    
    # Scarica e prepara il dataset
    training_data = download_and_prepare_dataset()
    
    # Mostra alcuni esempi
    show_sample_examples(training_data)
    
    print("\n🎯 Prossimi step:")
    print(f"1. Verifica il file: {OUTPUT_FILE}")
    print("2. Adatta MAX_SAMPLES se necessario")
    print("3. Esegui il training con:")
    print(f"   python test_training_lightweight.py --dataset {OUTPUT_FILE} --project_name tolkien_expert")
