"""Run a quick test of OllamaWrapper: prefer HTTP, fallback to CLI.

Usage:
  python scripts/run_wrapper_test.py --model gemma3:4b --prompt "Hello"

This script prints the response and exits.
"""
import argparse
from ollama_wrapper import OllamaWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma3:4b")
    p.add_argument("--prompt", default="Hello from wrapper test")
    args = p.parse_args()

    w = OllamaWrapper(base_url="http://localhost:11434/api", model_name=args.model)
    print("Using model:", args.model)
    resp = w.chat(args.prompt, timeout=60)
    print("Response:")
    print(resp)


if __name__ == "__main__":
    main()
