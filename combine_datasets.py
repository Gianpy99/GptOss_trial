"""
Crea un dataset combinato F1 + Tolkien per un modello multi-domain
"""
import json
from pathlib import Path

def combine_datasets():
    # Carica F1 dataset
    f1_file = Path("f1_training_data.json")
    tolkien_file = Path("tolkien_training_data.json")
    output_file = Path("combined_f1_tolkien_data.json")
    
    print("📥 Caricamento dataset F1...")
    with open(f1_file, 'r', encoding='utf-8') as f:
        f1_data = json.load(f)
    print(f"   ✅ {len(f1_data)} esempi F1")
    
    print("📥 Caricamento dataset Tolkien...")
    with open(tolkien_file, 'r', encoding='utf-8') as f:
        tolkien_data = json.load(f)
    
    # Normalizza il formato
    tolkien_normalized = []
    for item in tolkien_data:
        if isinstance(item, dict):
            instr = item.get('instruction') or item.get('prompt')
            resp = item.get('response') or item.get('completion')
            if instr and resp:
                tolkien_normalized.append({
                    'instruction': str(instr).strip(),
                    'response': str(resp).strip()
                })
    
    print(f"   ✅ {len(tolkien_normalized)} esempi Tolkien")
    
    # Combina
    combined = f1_data + tolkien_normalized
    print(f"\n📊 Dataset combinato: {len(combined)} esempi totali")
    print(f"   - F1: {len(f1_data)} esempi")
    print(f"   - Tolkien: {len(tolkien_normalized)} esempi")
    
    # Salva
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Salvato: {output_file}")
    print(f"\n🎯 Per trainare il modello ibrido:")
    print(f"   python test_training_lightweight.py {output_file} f1_tolkien_expert")
    print(f"\n   Tempo stimato: ~15-20 minuti")

if __name__ == "__main__":
    combine_datasets()
