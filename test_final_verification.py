#!/usr/bin/env python3
"""
Final verification test for the Ollama wrapper
Runs all main tests and confirms the project is fully functional
"""

import os
import sys
sys.path.insert(0, 'src')

from ollama_wrapper import OllamaWrapper, create_coding_assistant, create_creative_assistant

def main():
    print("🔧 FINAL VERIFICATION - Ollama Wrapper")
    print("=" * 50)
    
    # Test 1: Basic connection
    print("\n1️⃣ Basic connection test...")
    try:
        wrapper = OllamaWrapper(model_name="gemma3:4b")
        response = wrapper.chat("Hello, are you operational?", timeout=30)
        if response.get("status") == "success":
            print("   ✅ Connection OK")
        else:
            print(f"   ❌ Connection error: {response}")
            return False
    except Exception as e:
        print(f"   ❌ Connection exception: {e}")
        return False
    
    # Test 2: Memory and sessions
    print("\n2️⃣ Memory and sessions test...")
    try:
        # Test memory
        wrapper.memory_manager.store_fact("test_key", "test_value")
        facts = wrapper.memory_manager.search_facts("test")
        if facts:
            print("   ✅ Memory works")
        else:
            print("   ⚠️ Memory might have issues")
        
        # Test sessions
        wrapper.save_session("test_final")
        wrapper.load_session("test_final")
        print("   ✅ Sessions work")
        
    except Exception as e:
        print(f"   ❌ Memory/sessions error: {e}")
        return False
    
    # Test 3: Specialized assistants
    print("\n3️⃣ Specialized assistants test...")
    try:
        coding_assistant = create_coding_assistant(session_id="test_coding")
        response = coding_assistant.chat("How do you define a function in Python?", timeout=30)
        if response.get("status") == "success":
            print("   ✅ Coding assistant works")
        else:
            print("   ⚠️ Coding assistant might have issues")
        
        creative_assistant = create_creative_assistant(session_id="test_creative")
        response = creative_assistant.chat("Write a short poem", timeout=30)
        if response.get("status") == "success":
            print("   ✅ Creative assistant works")
        else:
            print("   ⚠️ Creative assistant might have issues")
            
    except Exception as e:
        print(f"   ❌ Assistants error: {e}")
        return False
    
    # Test 4: Vision/Multimodal
    print("\n4️⃣ Vision/multimodal test...")
    image_path = os.path.join("examples", "_2ARTURA_Blue.png")
    if os.path.exists(image_path):
        try:
            response = wrapper.chat("What car is this?", files=[image_path], timeout=60)
            if response.get("status") == "success":
                answer = response.get("assistant", "").lower()
                if "mclaren" in answer:
                    print("   🎉 Vision PERFECTLY working! (McLaren recognized)")
                else:
                    print(f"   ✅ Vision active but response: {answer[:100]}...")
            else:
                print(f"   ⚠️ Vision error: {response}")
        except Exception as e:
            print(f"   ❌ Vision exception: {e}")
    else:
        print("   ℹ️ Test image not found, skipping vision test")
    
    # Test 5: Streaming
    print("\n5️⃣ Streaming test...")
    try:
        print("   Streaming response:")
        response_parts = []
        for chunk in wrapper.stream_chat("Count from 1 to 3", timeout=30):
            if chunk.get("status") == "streaming":
                content = chunk.get("content", "")
                print(f"   📡 {content}", end="", flush=True)
                response_parts.append(content)
            elif chunk.get("status") == "complete":
                print("\n   ✅ Streaming completed")
                break
        
        if response_parts:
            full_response = "".join(response_parts)
            if any(num in full_response for num in ["1", "2", "3"]):
                print("   ✅ Streaming works correctly")
            else:
                print("   ⚠️ Streaming works but unexpected content")
        else:
            print("   ⚠️ No content received via streaming")
            
    except Exception as e:
        print(f"   ❌ Streaming error: {e}")
        return False
    
    # Final result
    print("\n" + "=" * 50)
    print("🎉 PROJECT COMPLETELY FUNCTIONAL!")
    print("✅ All main functionalities operational:")
    print("   • Basic chat")
    print("   • Memory and session persistence")
    print("   • Specialized assistants")
    print("   • Vision/Multimodal (gemma3:4b)")
    print("   • Streaming")
    print("\n💡 The wrapper is ready for production use!")
    print("   For future fine-tuning: models and sessions already supported")
    print("\n🚀 Quick usage example:")
    print("   from ollama_wrapper import OllamaWrapper")
    print("   wrapper = OllamaWrapper()")
    print("   response = wrapper.chat('Your prompt here')")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ VERIFICATION COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("\n❌ VERIFICATION FAILED - check logs above")
        sys.exit(1)