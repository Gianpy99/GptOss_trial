<!--
Guidance for AI coding agents working on this repository. Keep concise and specific to discoverable patterns.
-->
# Copilot / AI agent instructions for Ollama_wrapper

Summary
- This repository is a small Python utility wrapping a local Ollama server (default base URL: `http://localhost:11434/api`). The primary implementation is in `wrapper.py` which provides:
  - `OllamaWrapper` — high-level API for chat, streaming chat, multimodal attachments, session save/load, and memory backed by SQLite.
  - `MemoryManager` — SQLite-based conversation and facts storage (`ollama_memory.db`).
  - `OllamaCLIHelper` — thin safe subprocess wrapper around the `ollama` binary (optional).
  - Example factory functions: `create_coding_assistant`, `create_creative_assistant` and `interactive_repl`.

What to know up front
- Default model: `gemma3:8b-instruct` (constant `DEFAULT_MODEL`). Code expects a local Ollama HTTP API at `DEFAULT_BASE_URL`.
- Sessions are stored as JSON under the `ollama_sessions` directory; memories use a SQLite DB `ollama_memory.db` in the project root.
- Network calls use `requests` synchronously; streaming endpoints yield newline-delimited JSON lines and are parsed by `json.loads` per-line.

Files and important symbols
- `wrapper.py`: primary source of truth. Key classes/methods to reference in changes and tests:
  - `OllamaWrapper.__init__`, `chat`, `stream_chat`, `_build_messages`, `chat_with_files` (implemented as `chat` with `files`), `list_models`, `pull_model`, `show_model_info`, `save_session`, `load_session`.
  - `MemoryManager.store_message`, `get_conversation_history`, `store_fact`, `search_facts`.
  - `OllamaCLIHelper.run`, `pull`, `list`, `show`.
- `example.py`: demonstrates expected public API usage (note: some function names referenced here differ slightly from `wrapper.py` — prefer the definitions in `wrapper.py` when in doubt).

Project-specific conventions and patterns
- Session & memory handling:
  - Conversation history is reconciled by `OllamaWrapper._build_messages()` which prepends `system` prompt (if set) and previous messages from `MemoryManager` in chronological order.
  - When storing messages after chat/streaming, the wrapper first stores the user message then the assistant response.
  - Facts table is used for longer-term memory; `store_fact` upserts by key.
- Streaming parsing:
  - The stream-handling code expects either JSON lines with keys like `message`, `delta`, or `chunk`, or raw text lines. Agents modifying streaming logic must maintain tolerant parsing (fall back on raw chunks when JSON decoding fails).
- Files/attachments:
  - Files are embedded as base64 in the last message's `attachments` field. MIME hint is determined by file extension suffix only.

Build / run / debug commands (discoverable)
- No build system or dependencies manifest is present. The runtime requires Python 3.8+ and the `requests` package.
- Typical quick commands to run examples (Windows PowerShell):
  - Install dependency: `pip install requests`
  - Run demo script: `python .\wrapper.py` (the module has a `__main__` demo)
  - Run example usage: `python .\example.py`

Testing and validation hints for agents
- Add unit tests focusing on:
  - `MemoryManager` schema creation, store/get/list/search behaviors (use a temp SQLite file).
  - `OllamaWrapper._build_messages` with and without `system_prompt` and stored history.
  - `stream_chat` handling by mocking `requests.post(..., stream=True)` to yield JSON lines and raw text lines.
- Avoid network calls in unit tests; mock `requests.post` and `requests.get` to return controlled responses.

Edge cases observed (useful when editing/adding code)
- Streaming endpoints may return non-JSON chunks; keep fallback behavior.
- `pull_model` attempts `/pull` and reads streamed events; callers may request `stream=True` and expect a list of events.
- SQLite is used with `check_same_thread=False` and a connection per operation — tests and code should mimic this to avoid threading issues.

When editing the repository
- Prefer modifying `wrapper.py` (single file) and add tests under a `tests/` folder.
- Preserve the public API names as defined in `wrapper.py`. `example.py` may contain slightly different names — trust the implementation in `wrapper.py`.
- If adding dependencies, include a `requirements.txt` with pinned versions.

Examples to copy when changing behavior
- To create a low-temperature coding assistant:
  - `create_coding_assistant(session_id="coding")` — sets `temperature=0.2` and a system prompt to prefer precise code.
- To persist and load a session:
  - `wrapper.save_session("my-sess")` and `wrapper.load_session("my-sess")` — sessions saved to `ollama_sessions/my-sess.json`.

If you need more
- If behavior isn't clear from `wrapper.py`, run the demo `python .\wrapper.py` or `python .\example.py` and inspect runtime outputs. Ask for any missing conventions you want explicitly documented.

---
Please review and tell me if you want more detail about build steps, tests, or CI wiring.
