"""
This file is the original `wrapper.py` moved into the package layout.
Keep the implementation identical to the repository root version.
"""

# ...existing code...

import os
import json
import base64
import sqlite3
import requests
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Generator, Optional, Tuple, Union

# Constants
DEFAULT_BASE_URL = "http://localhost:11434/api"
DEFAULT_MODEL = "gemma3:4b"
SESSIONS_DIR = "ollama_sessions"
MEMORY_DB = "ollama_memory.db"
DEFAULT_STREAM_TIMEOUT = 120

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _encode_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token for English text."""
    return len(text) // 4

def _calculate_quality_metrics(response: str) -> Dict[str, Any]:
    """Calculate basic quality metrics for a response."""
    length = len(response)
    tokens = _estimate_tokens(response)
    sentences = len([s for s in response.split('.') if s.strip()])
    words = len(response.split())
    avg_word_length = sum(len(word) for word in response.split()) / max(words, 1)
    
    return {
        "length": length,
        "estimated_tokens": tokens,
        "sentences": sentences,
        "words": words,
        "avg_word_length": round(avg_word_length, 2)
    }

@dataclass
class ModelParameters:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    max_tokens: Optional[int] = 1024
    repeat_penalty: Optional[float] = None
    seed: Optional[int] = None
    num_ctx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}

@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

class MemoryManager:
    def __init__(self, db_path: str = MEMORY_DB):
        self.db_path = db_path
        self._ensure_schema()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        conn = self._conn()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY,
                    key TEXT UNIQUE,
                    value TEXT,
                    category TEXT,
                    timestamp TEXT
                );
                """
            )
        conn.close()

    def store_message(self, session_id: str, msg: Message):
        conn = self._conn()
        with conn:
            conn.execute(
                "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, msg.role, msg.content, msg.timestamp),
            )
        conn.close()

    def get_conversation_history(self, session_id: str, limit: int = 100) -> List[Message]:
        conn = self._conn()
        cursor = conn.execute(
            "SELECT role, content, timestamp FROM conversations WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        return [Message(role=r["role"], content=r["content"], timestamp=r["timestamp"]) for r in reversed(rows)]

    def clear_session(self, session_id: str):
        conn = self._conn()
        with conn:
            conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        conn.close()

    def store_fact(self, key: str, value: str, category: str = "general"):
        conn = self._conn()
        with conn:
            conn.execute(
                "INSERT INTO facts (key, value, category, timestamp) VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, category=excluded.category, timestamp=excluded.timestamp",
                (key, value, category, now_iso()),
            )
        conn.close()

    def get_fact(self, key: str) -> Optional[Tuple[str, str, str]]:
        conn = self._conn()
        cursor = conn.execute("SELECT key, value, category FROM facts WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return (row["key"], row["value"], row["category"])
        return None

    def list_facts(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        conn = self._conn()
        if category:
            cursor = conn.execute("SELECT key, value, category, timestamp FROM facts WHERE category = ?", (category,))
        else:
            cursor = conn.execute("SELECT key, value, category, timestamp FROM facts")
        rows = cursor.fetchall()
        conn.close()
        return [{"key": r["key"], "value": r["value"], "category": r["category"], "timestamp": r["timestamp"]} for r in rows]

    def search_facts(self, query: str, limit: int = 20) -> List[Dict[str, str]]:
        conn = self._conn()
        pattern = f"%{query}%"
        cursor = conn.execute(
            "SELECT key, value, category, timestamp FROM facts WHERE key LIKE ? OR value LIKE ? LIMIT ?",
            (pattern, pattern, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"key": r["key"], "value": r["value"], "category": r["category"], "timestamp": r["timestamp"]} for r in rows]

class OllamaWrapper:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = DEFAULT_MODEL,
        session_id: str = "default",
        parameters: Optional[ModelParameters] = None,
        memory_db_path: str = MEMORY_DB,
        prefer_cli: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.session_id = session_id
        self.parameters = parameters or ModelParameters()
        self.memory = MemoryManager(memory_db_path)
        self.prefer_cli = prefer_cli
        self.system_prompt: Optional[str] = None
        self.session_variables: Dict[str, Any] = {}
        safe_mkdir(SESSIONS_DIR)

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def set_session_variable(self, key: str, value: Any):
        self.session_variables[key] = value

    def get_session_variable(self, key: str, default: Any = None) -> Any:
        return self.session_variables.get(key, default)

    def list_session_variables(self) -> Dict[str, Any]:
        return self.session_variables.copy()

    def save_session(self, name: str) -> Dict[str, Any]:
        payload = {
            "model_name": self.model_name,
            "session_id": self.session_id,
            "parameters": self.parameters.to_dict(),
            "system_prompt": self.system_prompt,
            "session_variables": self.session_variables,
            "timestamp": now_iso(),
        }
        safe_mkdir(SESSIONS_DIR)
        path = os.path.join(SESSIONS_DIR, f"{name}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return {"status": "success", "file": path}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def load_session(self, name: str) -> Dict[str, Any]:
        path = os.path.join(SESSIONS_DIR, f"{name}.json")
        if not os.path.exists(path):
            return {"status": "error", "error": "session not found"}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.model_name = data.get("model_name", self.model_name)
            self.session_id = data.get("session_id", self.session_id)
            self.system_prompt = data.get("system_prompt", self.system_prompt)
            self.session_variables = data.get("session_variables", self.session_variables)
            if "parameters" in data:
                for k, v in data["parameters"].items():
                    if hasattr(self.parameters, k):
                        setattr(self.parameters, k, v)
            return {"status": "success", "session": name, "data": data}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def list_sessions(self) -> List[str]:
        if not os.path.exists(SESSIONS_DIR):
            return []
        files = [f[:-5] for f in os.listdir(SESSIONS_DIR) if f.endswith(".json")]
        return files

    def clear_session_context(self):
        self.memory.clear_session(self.session_id)
        self.session_variables.clear()
        self.system_prompt = None

    def list_models(self) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            try:
                resp = requests.get(f"{self.base_url}/tags", timeout=10)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                return {"error": str(e)}

    def pull_model(self, model_name: str, stream: bool = False) -> Dict[str, Any]:
        url = f"{self.base_url}/pull"
        try:
            with requests.post(url, json={"model": model_name}, stream=True, timeout=300) as r:
                r.raise_for_status()
                if stream:
                    events = []
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            events.append({"raw": line})
                    self.model_name = model_name
                    return {"status": "success", "events": events}
                else:
                    data = r.text
                    try:
                        parsed = json.loads(data)
                        self.model_name = model_name
                        return {"status": "success", "response": parsed}
                    except Exception:
                        self.model_name = model_name
                        return {"status": "success", "response_text": data}
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)}

    def show_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        model_to_show = model_name or self.model_name
        try:
            resp = requests.post(f"{self.base_url}/show", json={"model": model_to_show}, timeout=30)
            resp.raise_for_status()
            return {"status": "success", "model": model_to_show, "info": resp.json()}
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)}

    def _build_messages(self, user_message: str, include_history: bool = True) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if include_history:
            history = self.memory.get_conversation_history(self.session_id, limit=50)
            for m in history:
                messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": user_message})
        return messages

    def chat(
        self,
        message: str,
        include_history: bool = True,
        store_conversation: bool = True,
        files: Optional[List[str]] = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        start_time = time.time()
        url = f"{self.base_url}/chat"
        msgs = self._build_messages(message, include_history=include_history)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": msgs,
            "options": self.parameters.to_dict(),
            "stream": False,
        }

        if files:
            images = []
            for fpath in files:
                if not os.path.exists(fpath):
                    continue
                b64 = _encode_file_to_base64(fpath)
                images.append(b64)
            if images:
                payload["messages"][-1]["images"] = images

        # If configured to prefer CLI, skip HTTP and use CLI directly
        # BUT: Never use CLI when files/images are present (CLI doesn't support multimodal)
        if getattr(self, "prefer_cli", False) and not files:
            # Reuse robust CLI invocation logic (similar to HTTP fallback)
            try:
                import subprocess

                candidates = [
                    ["ollama", "run", self.model_name, "--format", "json", "--hidethinking", message],
                    ["ollama", "run", self.model_name, "--hidethinking", message],
                    ["ollama", "run", self.model_name, message],
                ]

                stdout_acc = None
                last_err = None
                for cmd in candidates:
                    try:
                        proc = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=timeout,
                        )
                    except FileNotFoundError:
                        last_err = "ollama binary not found"
                        break
                    except Exception as ex:
                        last_err = str(ex)
                        continue

                    out = proc.stdout if proc.stdout is not None else ""
                    err = proc.stderr if proc.stderr is not None else ""

                    parsed = None
                    try:
                        parsed = json.loads(out)
                    except Exception:
                        parsed = None

                    if parsed:
                        if isinstance(parsed, dict):
                            if "message" in parsed and isinstance(parsed["message"], dict):
                                stdout_acc = parsed["message"].get("content")
                            elif "choices" in parsed and parsed["choices"]:
                                ch = parsed["choices"][0]
                                stdout_acc = ch.get("message", {}).get("content") or ch.get("text")
                            elif "output" in parsed:
                                stdout_acc = parsed.get("output")
                            else:
                                stdout_acc = json.dumps(parsed)
                        else:
                            stdout_acc = str(parsed)
                    else:
                        if proc.returncode == 0 and out.strip():
                            stdout_acc = out.strip()

                    if stdout_acc:
                        assistant_text = stdout_acc
                        if store_conversation:
                            self.memory.store_message(self.session_id, Message("user", message))
                            self.memory.store_message(self.session_id, Message("assistant", assistant_text))
                        response_time = time.time() - start_time
                        quality_metrics = _calculate_quality_metrics(assistant_text)
                        return {
                            "status": "success", 
                            "assistant": assistant_text, 
                            "raw": {"cli_output": assistant_text},
                            "metrics": {
                                "response_time": round(response_time, 3),
                                "quality": quality_metrics
                            }
                        }
                return {"status": "error", "error": "cli_failed", "cli_error": last_err}
            except Exception as ex:
                return {"status": "error", "error": f"cli_exception: {ex}"}

        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            assistant_text = None
            if isinstance(result, dict):
                if "message" in result and isinstance(result["message"], dict):
                    assistant_text = result["message"].get("content")
                elif "choices" in result and isinstance(result["choices"], list) and result["choices"]:
                    ch = result["choices"][0]
                    assistant_text = ch.get("message", {}).get("content") or ch.get("text")
                elif "output" in result:
                    assistant_text = result.get("output")
            if assistant_text is None:
                assistant_text = json.dumps(result)[:2000]

            if store_conversation:
                self.memory.store_message(self.session_id, Message("user", message))
                self.memory.store_message(self.session_id, Message("assistant", assistant_text))

            response_time = time.time() - start_time
            quality_metrics = _calculate_quality_metrics(assistant_text)

            return {
                "status": "success", 
                "assistant": assistant_text, 
                "raw": result,
                "metrics": {
                    "response_time": round(response_time, 3),
                    "quality": quality_metrics
                }
            }
        except requests.RequestException as e:
            # HTTP API failed — attempt a robust CLI fallback using the `ollama` binary
            # BUT: Don't use CLI for multimodal requests (images not supported in CLI)
            if files:
                return {"status": "error", "error": f"HTTP API failed and CLI doesn't support images: {str(e)}"}
            
            try:
                import subprocess

                # candidate invocations to try; prefer JSON format to parse output
                candidates = [
                    ["ollama", "run", self.model_name, "--format", "json", "--hidethinking", message],
                    ["ollama", "run", self.model_name, "--hidethinking", message],
                    ["ollama", "run", self.model_name, message],
                    ["ollama", "chat", self.model_name, "--format", "json", "--hidethinking", message],
                    ["ollama", "chat", self.model_name, message],
                ]

                stdout_acc = None
                last_err = None
                for cmd in candidates:
                    try:
                        proc = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=timeout,
                        )
                    except FileNotFoundError:
                        last_err = "ollama binary not found"
                        break
                    except Exception as ex:
                        last_err = str(ex)
                        continue

                    out = proc.stdout if proc.stdout is not None else ""
                    err = proc.stderr if proc.stderr is not None else ""

                    # If the command returned JSON, try to parse assistant text
                    parsed = None
                    try:
                        parsed = json.loads(out)
                    except Exception:
                        parsed = None

                    if parsed:
                        # try common shapes
                        if isinstance(parsed, dict):
                            if "message" in parsed and isinstance(parsed["message"], dict):
                                stdout_acc = parsed["message"].get("content")
                            elif "choices" in parsed and parsed["choices"]:
                                ch = parsed["choices"][0]
                                stdout_acc = ch.get("message", {}).get("content") or ch.get("text")
                            elif "output" in parsed:
                                stdout_acc = parsed.get("output")
                            else:
                                stdout_acc = json.dumps(parsed)
                        else:
                            stdout_acc = str(parsed)
                    else:
                        # fallback: use raw stdout if non-empty
                        if proc.returncode == 0 and out.strip():
                            stdout_acc = out.strip()

                    if stdout_acc:
                        break
                    else:
                        last_err = (proc.returncode, out[:1000], err[:1000])

                if stdout_acc is not None:
                    assistant_text = stdout_acc
                    if store_conversation:
                        self.memory.store_message(self.session_id, Message("user", message))
                        self.memory.store_message(self.session_id, Message("assistant", assistant_text))
                    response_time = time.time() - start_time
                    quality_metrics = _calculate_quality_metrics(assistant_text)
                    return {
                        "status": "success", 
                        "assistant": assistant_text, 
                        "raw": {"cli_output": assistant_text},
                        "metrics": {
                            "response_time": round(response_time, 3),
                            "quality": quality_metrics
                        }
                    }
                else:
                    return {"status": "error", "error": str(e), "cli_error": last_err}
            except Exception as ex:
                return {"status": "error", "error": f"http_error: {e}; cli_error: {ex}"}
        except Exception as e:
            return {"status": "error", "error": f"parse_error: {e}"}

    def stream_chat(
        self,
        message: str,
        include_history: bool = True,
        timeout: int = DEFAULT_STREAM_TIMEOUT,
    ) -> Generator[str, None, Dict[str, Any]]:
        start_time = time.time()
        url = f"{self.base_url}/chat"
        msgs = self._build_messages(message, include_history=include_history)

        payload = {
            "model": self.model_name,
            "messages": msgs,
            "options": self.parameters.to_dict(),
            "stream": True,
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                collected = ""
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        chunk = line
                        collected += chunk
                        yield chunk
                        continue

                    chunk_text = ""
                    if isinstance(data, dict):
                        if "message" in data and isinstance(data["message"], dict):
                            chunk_text = data["message"].get("content", "")
                        elif "delta" in data:
                            chunk_text = data["delta"].get("content", "")
                        elif "chunk" in data:
                            chunk_text = str(data["chunk"])
                        else:
                            chunk_text = json.dumps(data)
                    else:
                        chunk_text = str(data)

                    if chunk_text:
                        collected += chunk_text
                        yield chunk_text

                self.memory.store_message(self.session_id, Message("user", message))
                self.memory.store_message(self.session_id, Message("assistant", collected))
                response_time = time.time() - start_time
                quality_metrics = _calculate_quality_metrics(collected)
                return {
                    "status": "success", 
                    "assistant": collected,
                    "metrics": {
                        "response_time": round(response_time, 3),
                        "quality": quality_metrics
                    }
                }
        except requests.RequestException as e:
            yield f"[stream_error] {e}"
            return {"status": "error", "error": str(e)}

    def store_memory(self, key: str, value: str, category: str = "general"):
        self.memory.store_fact(key, value, category)

    def recall_memory(self, key: str) -> Optional[Tuple[str, str, str]]:
        return self.memory.get_fact(key)

    def list_memories(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        return self.memory.list_facts(category)

    def search_memories(self, query: str, limit: int = 20) -> List[Dict[str, str]]:
        return self.memory.search_facts(query, limit=limit)

class OllamaCLIHelper:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

    def run(self, *args: str, timeout: int = 30) -> Dict[str, Any]:
        try:
            import subprocess

            proc = subprocess.run(["ollama", *args], capture_output=True, text=True, timeout=timeout)
            return {"status": "success", "stdout": proc.stdout, "stderr": proc.stderr, "code": proc.returncode}
        except FileNotFoundError:
            return {"status": "error", "error": "ollama binary not found in PATH"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def pull(self, model: Optional[str] = None):
        return self.run("pull", model or self.model_name)

    def list(self):
        return self.run("list")

    def show(self, model: Optional[str] = None):
        return self.run("show", model or self.model_name)

def create_coding_assistant(session_id: str = "coding") -> OllamaWrapper:
    params = ModelParameters(temperature=0.2, top_p=0.9, max_tokens=2048)
    w = OllamaWrapper(model_name=DEFAULT_MODEL, session_id=session_id, parameters=params)
    w.set_system_prompt("You are a precise coding assistant. Provide well-commented, tested code examples.")
    return w

def create_creative_assistant(session_id: str = "creative") -> OllamaWrapper:
    params = ModelParameters(temperature=0.95, top_p=0.95, max_tokens=2048)
    w = OllamaWrapper(model_name=DEFAULT_MODEL, session_id=session_id, parameters=params)
    w.set_system_prompt("You are a creative writing assistant. Be imaginative and playful while staying coherent.")
    return w

def interactive_repl(wrapper: OllamaWrapper):
    print("Interactive Ollama REPL (type /exit to quit).")
    print("Hint: prefix commands with / ; otherwise text is sent to model.")
    while True:
        try:
            raw = input(f"{wrapper.model_name}:{wrapper.session_id}> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting REPL.")
            break

        if not raw:
            continue

        if raw.startswith("/"):
            parts = raw.split()
            cmd = parts[0].lower()
            if cmd == "/exit":
                break
            elif cmd == "/save" and len(parts) >= 2:
                print(wrapper.save_session(parts[1]))
            elif cmd == "/load" and len(parts) >= 2:
                print(wrapper.load_session(parts[1]))
            elif cmd == "/sessions":
                print(wrapper.list_sessions())
            elif cmd == "/mem" and len(parts) >= 2:
                sub = parts[1]
                if sub == "set" and len(parts) >= 4:
                    key = parts[2]
                    value = " ".join(parts[3:])
                    wrapper.store_memory(key, value)
                    print("ok")
                elif sub == "get" and len(parts) >= 3:
                    print(wrapper.recall_memory(parts[2]))
                elif sub == "list":
                    cat = parts[2] if len(parts) >= 3 else None
                    print(wrapper.list_memories(cat))
                else:
                    print("Unknown mem subcommand")
            elif cmd == "/stream" and len(parts) >= 2:
                text = " ".join(parts[1:])
                print("streaming...")
                buffer = ""
                for chunk in wrapper.stream_chat(text):
                    print(chunk, end="", flush=True)
                    buffer += chunk
                print("\n--- done ---")
            else:
                print("Unknown or malformed command.")
        else:
            resp = wrapper.chat(raw)
            if resp.get("status") == "success":
                print(resp.get("assistant"))
            else:
                print("Error:", resp.get("error"))

if __name__ == "__main__":
    w = create_coding_assistant("demo")
    print("Demo: simple chat")
    r = w.chat("Hello! Summarize the responsibilities of an embedded systems engineer in 3 bullets.")
    print(r.get("assistant") if r.get("status") == "success" else r)

    print("\nDemo: store and recall memory")
    w.store_memory("fav_language", "Python", category="preferences")
    print("Recall:", w.recall_memory("fav_language"))
