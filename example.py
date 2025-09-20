from ollama_wrapper import OllamaWrapper, OllamaCLI  # assuming you saved the file as ollama_wrapper.py

# Initialize wrapper with Gemma 3 (default is already gemma3:8b-instruct)
ollama = OllamaWrapper()

# --- 1. Simple text chat ---
print("=== Simple Chat ===")
response = ollama.chat("Explain quantum entanglement in simple terms.")
print(response)


# --- 2. Chat with context (system message) ---
print("\n=== Chat with Context ===")
response = ollama.chat("What are the laws of robotics?", context="You are Isaac Asimov.")
print(response)


# --- 3. Streaming chat (token by token) ---
print("\n=== Streaming Chat ===")
for token in ollama.chat_stream("Write a haiku about black holes."):
    print(token, end="", flush=True)
print("\n")


# --- 4. Multimodal chat (text + images) ---
print("\n=== Chat with Image ===")
response = ollama.chat_with_images("Describe this picture.", ["example.jpg"])
print(response)


# --- 5. Memory: store and retrieve facts ---
print("\n=== Memory Store & Retrieve ===")
ollama.store_memory("favorite_drink", "espresso", category="preferences")
ollama.store_memory("favorite_drink", "green tea", category="preferences")

facts = ollama.retrieve_memory("favorite_drink")
for f in facts:
    print(f)


# --- 6. CLI integration ---
print("\n=== CLI Integration ===")
cli = OllamaCLI()
print(cli.list_models())
print(cli.show_model_info())
