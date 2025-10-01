# 🔐 Setup Hugging Face Authentication

## Perché serve?

Gemma e Llama sono modelli **"gated"** (protetti da licenza). Google/Meta richiedono:
- Account Hugging Face
- Accettazione termini d'uso
- Token di autenticazione

## 📝 Setup (5 minuti)

### Step 1: Crea Account Hugging Face
1. Vai su https://huggingface.co/join
2. Registrati (gratis)

### Step 2: Accetta Licenza Gemma
1. Vai su https://huggingface.co/google/gemma-2-2b
2. Clicca **"Agree and access repository"**
3. Ripeti per https://huggingface.co/google/gemma-2-9b

### Step 3: Crea Token
1. Vai su https://huggingface.co/settings/tokens
2. Clicca **"New token"**
3. Nome: `ollama-finetuning`
4. Type: **Read**
5. Copia il token: `hf_xxxxxxxxxxxxxxxxx`

### Step 4: Autenticazione

**Metodo A: CLI (persistente)**
```powershell
# Installa CLI
pip install -U huggingface_hub[cli]

# Login (salva token permanentemente)
huggingface-cli login

# Incolla il token quando richiesto
```

**Metodo B: Environment Variable (sessione)**
```powershell
# Windows PowerShell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"

# Poi esegui il training nella stessa sessione
python finetuning_workflow.py train ...
```

**Metodo C: Python Script**
```python
from huggingface_hub import login
login(token="hf_xxxxxxxxxxxxxxxxx")
```

## ✅ Verifica Setup

```powershell
# Test autenticazione
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b'); print('✓ Auth OK!')"
```

Se vedi `✓ Auth OK!` → sei pronto!

Se vedi `GatedRepoError` → ricontrolla i 4 step sopra.

## 🚀 Ora puoi fare Fine-Tuning

```powershell
# Gemma 2B (più veloce)
python finetuning_workflow.py train `
  --dataset "Vadera007/Formula_1_Dataset" `
  --project "f1_expert" `
  --model "gemma3:4b" `
  --type "f1"

# Gemma 9B (più potente)
python finetuning_workflow.py train `
  --dataset "..." `
  --project "..." `
  --model "gemma3:8b" `
  --type "f1"
```

## 🔄 Mapping Modelli

| Ollama Model | Hugging Face Model | Size | Auth Required |
|--------------|-------------------|------|---------------|
| `gemma3:4b`  | google/gemma-2-2b | 2B   | ✅ Sì         |
| `gemma3:8b`  | google/gemma-2-9b | 9B   | ✅ Sì         |
| `llama3:8b`  | meta-llama/Llama-3.1-8B | 8B | ✅ Sì    |
| `phi2`       | microsoft/phi-2   | 2.7B | ❌ No         |

## 💡 Alternative Senza Auth

Se non vuoi autenticarti, usa **Phi-2**:

```powershell
# Download phi-2 in Ollama (se non ce l'hai)
ollama pull phi

# Fine-tuning con Phi-2
python finetuning_workflow.py train `
  --dataset "Vadera007/Formula_1_Dataset" `
  --project "f1_expert" `
  --model "phi2" `
  --type "f1"
```

**NOTA**: L'adapter Phi-2 NON funziona con Gemma! Devi usare lo stesso modello.

## 🔧 Troubleshooting

### "401 Unauthorized"
→ Token non valido o non hai accettato la licenza

### "Rate limit exceeded"
→ Aspetta qualche minuto, poi riprova

### "Token expired"
→ Crea nuovo token su https://huggingface.co/settings/tokens

### "Model not found"
→ Assicurati di aver accettato la licenza del modello specifico

## 📚 Link Utili

- Hugging Face Tokens: https://huggingface.co/settings/tokens
- Gemma 2B: https://huggingface.co/google/gemma-2-2b
- Gemma 9B: https://huggingface.co/google/gemma-2-9b
- Llama 3.1: https://huggingface.co/meta-llama/Llama-3.1-8B
- Phi-2 (no auth): https://huggingface.co/microsoft/phi-2

## ✨ Dopo l'Autenticazione

Il token viene salvato in:
- Windows: `C:\Users\<user>\.cache\huggingface\token`
- Linux/Mac: `~/.cache/huggingface/token`

Non serve rifare login ogni volta! 🎉
