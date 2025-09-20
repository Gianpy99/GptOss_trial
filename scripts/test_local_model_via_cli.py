#!/usr/bin/env python3
"""Test a local Ollama model via the `ollama` CLI.

Usage:
  python scripts/test_local_model_via_cli.py --model <model_name> [--prompt "Hello"]

The script will try several common CLI invocation patterns until one succeeds.
"""
import argparse
import shutil
import subprocess
import sys


def find_ollama():
    return shutil.which("ollama")


def try_commands(ollama_bin, model, prompt):
    # common variants to try
    candidates = [
        [ollama_bin, "run", model, "--prompt", prompt],
        [ollama_bin, "run", model, prompt],
        [ollama_bin, "chat", model, "--prompt", prompt],
        [ollama_bin, "chat", model, prompt],
        [ollama_bin, "eval", model, "--prompt", prompt],
    ]

    for cmd in candidates:
        try:
            print(f"Trying: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except FileNotFoundError:
            print("ollama binary not found while executing candidate.")
            return False, None
        except subprocess.TimeoutExpired:
            print("Command timed out")
            continue

        if proc.returncode == 0 and proc.stdout.strip():
            return True, proc.stdout

        # if stderr contains useful info, show for debugging
        print(f"returncode={proc.returncode} stdout=(trimmed) {proc.stdout[:200]!r} stderr=(trimmed) {proc.stderr[:200]!r}")

    return False, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Model name as known to ollama (folder name under ~/.ollama/models)")
    p.add_argument("--prompt", default="Hello from test", help="Prompt to send to the model")
    args = p.parse_args()

    ollama = find_ollama()
    if not ollama:
        print("Error: 'ollama' binary not found in PATH. Install Ollama or add it to PATH.")
        sys.exit(2)

    ok, out = try_commands(ollama, args.model, args.prompt)
    if ok:
        print("\n=== SUCCESS: output ===\n")
        print(out)
        sys.exit(0)
    else:
        print("\nNo candidate invocation succeeded. Check the model name or the installed ollama CLI version.")
        sys.exit(3)


if __name__ == "__main__":
    main()
