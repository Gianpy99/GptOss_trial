#!/usr/bin/env python3
"""
Demonstration script for the Ollama wrapper.
Practical examples of using the local model.
"""

import sys
import os

# Add src to path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ollama_wrapper import OllamaWrapper, create_coding_assistant, create_creative_assistant

def demo_simple_chat():
    """Simple chat example"""
    print("🗨️  Demo: Simple Chat")
    print("-" * 30)
    
    wrapper = OllamaWrapper()
    
    questions = [
        "Hello! Can you introduce yourself briefly?",
        "What is the capital of Italy?",
        "Explain what lists are in Python in a simple way"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        response = wrapper.chat(question, timeout=45)
        if response.get("status") == "success":
            print(f"   Response: {response.get('assistant')}")
        else:
            print(f"   Error: {response.get('error')}")
    
    print("\n" + "="*50)

def demo_programming_assistant():
    """Programming assistant example"""
    print("💻 Demo: Programming Assistant")
    print("-" * 35)
    
    coding = create_coding_assistant("programming_demo")
    
    requests = [
        "Write a Python function that calculates the Fibonacci sequence",
        "How are exceptions handled in Python?",
        "Explain the difference between list and tuple in Python"
    ]
    
    for i, request in enumerate(requests, 1):
        print(f"\n{i}. Request: {request}")
        response = coding.chat(request, timeout=60)
        if response.get("status") == "success":
            print(f"   Response: {response.get('assistant')[:300]}...")
        else:
            print(f"   Error: {response.get('error')}")
    
    print("\n" + "="*50)

def demo_streaming():
    """Streaming chat example"""
    print("🌊 Demo: Streaming Chat")
    print("-" * 25)
    
    wrapper = OllamaWrapper()
    
    print("\nQuestion: Tell a short fantasy story")
    print("Streaming response:")
    print("-" * 40)
    
    try:
        for chunk in wrapper.stream_chat("Tell a short fantasy story in 3 sentences"):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 40)
        print("✓ Streaming completed")
    except Exception as e:
        print(f"\n❌ Streaming error: {e}")
    
    print("\n" + "="*50)

def demo_memory():
    """Memory system usage example"""
    print("🧠 Demo: Memory System")
    print("-" * 28)
    
    wrapper = OllamaWrapper(session_id="demo_memory")
    
    # Store some information
    print("\n1. Storing information...")
    wrapper.store_memory("preferred_language", "Python", "preferences")
    wrapper.store_memory("current_project", "Ollama Wrapper", "work")
    wrapper.store_memory("hobby", "Machine Learning", "personal")
    
    # Retrieve information
    print("2. Retrieving stored information:")
    facts = wrapper.list_memories()
    for fact in facts:
        print(f"   - {fact['key']}: {fact['value']} ({fact['category']})")
    
    # Search in memory
    print("\n3. Memory search for 'Python':")
    search_results = wrapper.search_memories("Python")
    for result in search_results:
        print(f"   - Found: {result['key']} = {result['value']}")
    
    # Test conversation with memory
    print("\n4. Chat with persistent memory:")
    wrapper.chat("My preferred language is Python")
    wrapper.chat("I'm working on an Ollama wrapper")
    
    # New conversation that should remember
    response = wrapper.chat("What did we talk about before?")
    if response.get("status") == "success":
        print(f"   Response: {response.get('assistant')[:200]}...")
    
    print("\n" + "="*50)

def demo_sessions():
    """Session management example"""
    print("💾 Demo: Session Management")
    print("-" * 27)
    
    # Create a session with specific configuration
    wrapper = OllamaWrapper(session_id="demo_sessions")
    wrapper.set_system_prompt("You are a mathematics specialist assistant. Always respond precisely and with examples.")
    
    # Chat with the configured session
    print("\n1. Chat with custom system prompt:")
    response = wrapper.chat("Explain Pythagoras' theorem")
    if response.get("status") == "success":
        print(f"   Response: {response.get('assistant')[:200]}...")
    
    # Save the session
    print("\n2. Saving session...")
    save_result = wrapper.save_session("mathematics_session")
    if save_result.get("status") == "success":
        print("   ✓ Session saved successfully")
    
    # List available sessions
    print("\n3. Available sessions:")
    sessions = wrapper.list_sessions()
    for session in sessions:
        print(f"   - {session}")
    
    print("\n" + "="*50)

def main():
    """Runs all demos"""
    print("🚀 Ollama Wrapper Demo")
    print("=" * 50)
    print("This script demonstrates the main wrapper functionalities")
    print("=" * 50)
    
    try:
        demo_simple_chat()
        demo_streaming()
        demo_memory()
        demo_sessions()
        demo_programming_assistant()
        
        print("🎉 All demos completed!")
        print("\n📖 For more information, check README.md")
        print("💡 Start using the wrapper in your projects!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()