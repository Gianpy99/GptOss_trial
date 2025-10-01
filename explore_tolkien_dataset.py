"""
Esplora il dataset Tolkien per capire la sua struttura.
"""

import pandas as pd
from datasets import load_dataset

print("🔍 Esplorazione Dataset Tolkien\n")

try:
    # Prova con la libreria datasets di HuggingFace
    print("📥 Caricamento con datasets library...")
    dataset = load_dataset("siddharth-magesh/llm-tolkien")
    
    print(f"\n✅ Dataset caricato!")
    print(f"📊 Splits disponibili: {list(dataset.keys())}")
    
    # Mostra info sul train split
    train_dataset = dataset['train']
    print(f"\n📈 Train set: {len(train_dataset)} esempi")
    print(f"🔑 Features: {train_dataset.features}")
    
    # Mostra i primi esempi
    print(f"\n📝 Primi 3 esempi (RAW):\n")
    for i in range(min(3, len(train_dataset))):
        print(f"--- Esempio {i+1} ---")
        example = train_dataset[i]
        for key, value in example.items():
            if isinstance(value, list):
                print(f"{key}: [lista con {len(value)} elementi]")
                if len(value) > 0:
                    print(f"  Primi 10: {value[:10]}")
            else:
                print(f"{key}: {value}")
        print()
    
    # Prova a decodificare i token se possibile
    if 'input_ids' in train_dataset.features:
        print("\n🔤 Tentativo di decodifica token...")
        try:
            from transformers import AutoTokenizer
            
            # Prova diversi tokenizer comuni
            tokenizers_to_try = [
                "google/gemma-2-2b-it",
                "meta-llama/Llama-2-7b-hf",
                "gpt2"
            ]
            
            for tokenizer_name in tokenizers_to_try:
                try:
                    print(f"\n🔍 Provo con tokenizer: {tokenizer_name}")
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    
                    # Decodifica il primo esempio
                    decoded = tokenizer.decode(train_dataset[0]['input_ids'])
                    print(f"✅ Testo decodificato (primi 200 caratteri):")
                    print(decoded[:200] + "...")
                    break
                except Exception as e:
                    print(f"❌ Fallito: {e}")
                    continue
                    
        except ImportError:
            print("⚠️  transformers non installato, impossibile decodificare")

except Exception as e:
    print(f"\n❌ Errore: {e}")
    print("\n💡 Provo con il metodo alternativo (pandas)...")
    
    try:
        splits = {
            'train': 'data/train-00000-of-00001-b3b3f56c16c8ea71.parquet',
            'test': 'data/test-00000-of-00001-3363aebce64126f2.parquet'
        }
        df = pd.read_parquet("hf://datasets/siddharth-magesh/llm-tolkien/" + splits["train"])
        
        print(f"✅ Dataset caricato con pandas: {len(df)} righe")
        print(f"🔑 Colonne: {df.columns.tolist()}")
        print("\n📊 Info dataset:")
        print(df.info())
        print("\n📝 Primi esempi:")
        print(df.head())
        
    except Exception as e2:
        print(f"❌ Anche il metodo alternativo è fallito: {e2}")
