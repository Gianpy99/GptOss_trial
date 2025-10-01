"""
UI GPU SENZA Quantizzazione
Float16 puro - dovrebbe funzionare su 6GB
"""

import os
print("🚀 F1 Expert UI - GPU Float16 (No Quantization)")
print()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

MODEL_PATH = "./fine_tuned_models/f1_expert_merged"
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 0:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"✓ pad_token_id={tokenizer.pad_token_id}")
print()

print("Loading model (float16, ~40 seconds)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # NO quantization
    device_map="auto",  # Automatic GPU placement
    token=HF_TOKEN
)
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()
print("✓ Model loaded on GPU")
print()

def chat(message, temperature=0.7, max_tokens=150):
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(message):
        response = response[len(message):].strip()
    return response

import gradio as gr

examples = [
    "Tell me about Lewis Hamilton",
    "McLaren lap times",
    "Monaco GP winner"
]

with gr.Blocks(title="F1 Expert", theme="soft") as demo:
    gr.Markdown("# 🏎️ F1 Expert (GPU Float16)")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=400)
            msg = gr.Textbox(placeholder="Ask about F1...")
            submit = gr.Button("Send")
            
        with gr.Column(scale=1):
            temp = gr.Slider(0.1, 1.0, 0.7, label="Temperature")
            tokens = gr.Slider(50, 300, 150, label="Length")
            gr.Examples(examples, msg)
    
    def respond(message, history, temperature, max_tokens):
        if not message:
            return history, ""
        try:
            reply = chat(message, temperature, max_tokens)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
        except Exception as e:
            history.append({"role": "assistant", "content": f"Error: {e}"})
        return history, ""
    
    msg.submit(respond, [msg, chatbot, temp, tokens], [chatbot, msg])
    submit.click(respond, [msg, chatbot, temp, tokens], [chatbot, msg])

print("Starting UI on http://localhost:7863")
demo.launch(server_port=7863, inbrowser=True)
