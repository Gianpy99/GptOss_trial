#!/usr/bin/env python3
"""
Examples of Ollama wrapper usage.
Demonstrates main features in a simple way.
"""

from ollama_wrapper import OllamaWrapper, create_coding_assistant, create_creative_assistant

def main():
    print("🚀 Ollama Wrapper Usage Examples")
    print("=" * 40)
    
    # Example 1: Simple chat
    print("\n1. Simple Chat")
    print("-" * 20)
    wrapper = OllamaWrapper()
    response = wrapper.chat("Explain object-oriented programming in 2 lines")
    if response.get("status") == "success":
        print(f"Response: {response.get('assistant')}")
    else:
        print(f"Error: {response.get('error')}")

    # Example 2: Streaming chat
    print("\n2. Streaming Chat")
    print("-" * 20)
    print("Question: Write a haiku about computers")
    print("Response: ", end="")
    try:
        for chunk in wrapper.stream_chat("Write a haiku about computers"):
            print(chunk, end="", flush=True)
        print()  # New line at the end
    except Exception as e:
        print(f"Streaming error: {e}")

    # Example 3: Programming assistant
    print("\n3. Programming Assistant")
    print("-" * 30)
    coding = create_coding_assistant("coding_example")
    response = coding.chat("How do you create a list in Python?")
    if response.get("status") == "success":
        print(f"Technical response: {response.get('assistant')[:200]}...")
    else:
        print(f"Error: {response.get('error')}")

    # Example 4: Memory
    print("\n4. Memory System")
    print("-" * 22)
    wrapper.store_memory("language", "Python", "preferences")
    wrapper.store_memory("framework", "FastAPI", "web")
    
    # List memories
    facts = wrapper.list_memories()
    print(f"Stored facts: {len(facts)}")
    for fact in facts[-2:]:  # Show last 2
        print(f"  - {fact['key']}: {fact['value']} ({fact['category']})")

    # Example 5: Sessions
    print("\n5. Session Management")
    print("-" * 22)
    wrapper.set_system_prompt("You are an assistant that always responds with enthusiasm!")
    save_result = wrapper.save_session("example_session")
    if save_result.get("status") == "success":
        print("✓ Session saved successfully")
        
        # List sessions
        sessions = wrapper.list_sessions()
        print(f"Available sessions: {sessions}")
    else:
        print(f"Error saving: {save_result}")

    print("\n" + "=" * 40)
    print("✅ Examples completed!")
    print("💡 Now you can use the wrapper in your projects!")

if __name__ == "__main__":
    main()
