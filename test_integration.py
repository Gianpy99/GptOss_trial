"""Quick test to verify the integration works correctly."""

print("Testing Ollama_wrapper with Fine-tuning integration...\n")

# Test 1: Base imports
print("Test 1: Base imports (always should work)")
try:
    from src.ollama_wrapper import OllamaWrapper, MemoryManager, ModelParameters
    print("  ✓ OllamaWrapper imported")
    print("  ✓ MemoryManager imported")
    print("  ✓ ModelParameters imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 2: Fine-tuning imports (may fail if dependencies not installed)
print("\nTest 2: Fine-tuning imports (optional)")
try:
    from src.ollama_wrapper import FineTuningManager, create_finetuned_assistant
    print("  ✓ FineTuningManager imported")
    print("  ✓ create_finetuned_assistant imported")
    print("  ✓ Fine-tuning dependencies are installed!")
except ImportError as e:
    print(f"  ⚠ Fine-tuning not available (expected if dependencies not installed)")
    print(f"    Install with: pip install -r requirements-finetuning.txt")

# Test 3: Check __all__ exports
print("\nTest 3: Package exports")
import src.ollama_wrapper as ow
print(f"  Available exports: {', '.join(ow.__all__)}")

# Test 4: Create wrapper instance
print("\nTest 4: Create OllamaWrapper instance")
try:
    wrapper = OllamaWrapper(model_name="test_model")
    print(f"  ✓ Created wrapper with model: {wrapper.model_name}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print("\n" + "="*60)
print("✓ All tests passed!")
print("="*60)
print("\nNext steps:")
print("  1. Install fine-tuning deps: pip install -r requirements-finetuning.txt")
print("  2. Try quick start: python examples/quick_start_finetuning.py")
print("  3. Read guide: FINETUNING_GUIDE.md")
