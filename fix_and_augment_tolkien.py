"""
Fixa un file JSON malformato `tolkien_training_data.json` e genera due file:
- `tolkien_training_data_fixed.json` (JSON corretto, normalized)
- `tolkien_training_data_augmented.json` (aggiunge varianti per data augmentation)

Il metodo è conservativo: sostituisce i campi `prompt`/`completion` con
`instruction`/`response`, inserisce virgole mancanti tra oggetti
e crea varianti automatiche (summarize, one-sentence, 3-bullets).
"""
import json
import re
from pathlib import Path


def load_and_recover(path: Path):
    text = path.read_text(encoding='utf-8')

    # 1) Replace prompt/completion keys with instruction/response
    text = text.replace('"prompt"', '"instruction"')
    """
    Fixa un file JSON malformato `tolkien_training_data.json` e genera due file:
    - `tolkien_training_data_fixed.json` (JSON corretto, normalized)
    - `tolkien_training_data_augmented.json` (aggiunge varianti per data augmentation)

    Il metodo è conservativo: sostituisce i campi `prompt`/`completion` con
    `instruction`/`response`, inserisce virgole mancanti tra oggetti
    e crea varianti automatiche (summarize, one-sentence, 3-bullets).
    """
    import json
    import re
    from pathlib import Path


    def load_and_recover(path: Path):
        text = path.read_text(encoding='utf-8')

        # 1) Replace prompt/completion keys with instruction/response
        text = text.replace('"prompt"', '"instruction"')
        text = text.replace('"completion"', '"response"')

        # 2) Insert missing commas between objects: '}{' across newlines
        #    Replace patterns like '}{' with '},\n{'
        text = re.sub(r"}\s*\{", "},\n{", text)

        # 3) Ensure the array has commas between entries, then load
        try:
            data = json.loads(text)
            return data
        except Exception as e:
            # Try to be more tolerant: remove trailing commas and reattempt
            cleaned = re.sub(r",\s*\]", "]", text)
            try:
                data = json.loads(cleaned)
                return data
            except Exception as e2:
                print("Failed to parse JSON after recovery attempts:", e2)
                raise


    def normalize_examples(raw_examples):
        examples = []
        for item in raw_examples:
            if not isinstance(item, dict):
                continue
            # Normalize keys: either instruction/response or prompt/response
            instr = item.get('instruction') or item.get('prompt')
            resp = item.get('response') or item.get('completion')
            if instr and resp:
                examples.append({
                    'instruction': str(instr).strip(),
                    'response': str(resp).strip()
                })
            else:
                # Skip malformed entries
                continue
        return examples


    def first_sentence(text: str):
        # Very simple sentence splitter
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return parts[0] if parts else text


    def to_bullets(text: str, n=3):
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        bullets = [p.strip() for p in parts if p.strip()][:n]
        if not bullets:
            bullets = [text.strip()[:120]]
        return bullets


    def augment_examples(examples, variants_per_example=3):
        augmented = []
        for ex in examples:
            # 1) Summarize -> use truncated response
            summary = ex['response']
            summary_short = summary
            if len(summary_short) > 200:
                summary_short = summary_short[:200].rsplit(' ', 1)[0] + '...'
            augmented.append({
                'instruction': f"Summarize: {ex['instruction']}",
                'response': summary_short
            })

            # 2) One-sentence
            one = first_sentence(ex['response'])
            augmented.append({
                'instruction': f"In one sentence: {ex['instruction']}",
                'response': one
            })

            # 3) 3-bullets
            bullets = to_bullets(ex['response'], n=3)
            bullet_text = '\n'.join(['- ' + b for b in bullets])
            augmented.append({
                'instruction': f"List 3 facts about: {ex['instruction']}",
                'response': bullet_text
            })

        return augmented


    def main():
        src = Path('tolkien_training_data.json')
        fixed_out = Path('tolkien_training_data_fixed.json')
        aug_out = Path('tolkien_training_data_augmented.json')
        print('🔧 Loading and recovering', src)
        try:
            raw_text_preview = src.read_text(encoding='utf-8')[:200]
            print('📄 Source file preview (200 chars):')
            print(raw_text_preview)
        except Exception as e:
            print('⚠️ Cannot read source file:', e)

        try:
            raw = load_and_recover(src)
            examples = normalize_examples(raw)
            print(f'✅ Parsed examples: {len(examples)}')

            # Save normalized fixed file (overwrite original safe copy)
            fixed_out.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding='utf-8')
            print('💾 Wrote fixed file:', fixed_out)

            # Augment
            augmented = augment_examples(examples)
            combined = examples + augmented
            print(f'✨ Total after augmentation: {len(combined)} examples')

            aug_out.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding='utf-8')
            print('💾 Wrote augmented file:', aug_out)
        except Exception as e:
            print('❌ Error during processing:', type(e).__name__, e)


    if __name__ == '__main__':
        main()
