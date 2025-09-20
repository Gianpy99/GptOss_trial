import os
import requests
import pytest

from ollama_wrapper import OllamaWrapper


def is_ollama_available(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/models", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
def test_chat_with_local_ollama():
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api")
    if not is_ollama_available(base_url):
        pytest.skip(f"Local Ollama HTTP API not available at {base_url}")

    # Use the default model name (the wrapper will use the local Ollama models)
    w = OllamaWrapper(base_url=base_url)

    # Simple ping chat — many Ollama models accept a short prompt
    resp = w.chat("Hello from integration test", timeout=30)
    assert isinstance(resp, dict)
    assert resp.get("status") in ("success",)
    assert "assistant" in resp
