#!/usr/bin/env python3
"""
Quick test of the Ollama wrapper to verify it works correctly.
"""

import sys
import os

# Add src to path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from ollama_wrapper import OllamaWrapper
    print("✓ Import successful")
    
    # Test connection and list models
    print("\n=== Ollama Connection Test ===")
    wrapper = OllamaWrapper()
    
    print(f"Base URL: {wrapper.base_url}")
    print(f"Model: {wrapper.model_name}")
    
    # List available models
    print("\n=== Available Models List ===")
    models = wrapper.list_models()
    if "error" in models:
        print(f"❌ Error retrieving models: {models['error']}")
    else:
        print("✓ Available models:")
        if "models" in models:
            for model in models["models"][:3]:  # Show only first 3
                print(f"  - {model.get('name', 'N/A')}")
        else:
            print(f"  {models}")
    
    # Simple chat test
    print("\n=== Simple Chat Test ===")
    try:
        response = wrapper.chat("Hello, can you respond briefly in English?", timeout=30)
        if response.get("status") == "success":
            print("✓ Chat works!")
            print(f"Response: {response.get('assistant', 'N/A')[:100]}...")
        else:
            print(f"❌ Chat error: {response}")
    except Exception as e:
        print(f"❌ Exception during chat: {e}")
    
    print("\n=== Test Completed ===")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Generic error: {e}")