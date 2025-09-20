import os
import tempfile
from ollama_wrapper import MemoryManager, Message

def test_memory_store_and_retrieve():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        mm = MemoryManager(db_path=path)
        mm.store_message('sess1', Message('user', 'hello'))
        hist = mm.get_conversation_history('sess1')
        assert len(hist) == 1
        assert hist[0].content == 'hello'
    finally:
        os.remove(path)
