import unittest
from src.wrapper import MemoryManager

class TestMemoryManager(unittest.TestCase):

    def setUp(self):
        self.memory_manager = MemoryManager(db_path=':memory:')  # Use in-memory database for testing

    def test_store_message(self):
        self.memory_manager.store_message("user", "Hello, how are you?")
        history = self.memory_manager.get_conversation_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['role'], 'user')
        self.assertEqual(history[0]['content'], "Hello, how are you?")

    def test_store_fact(self):
        self.memory_manager.store_fact("key1", "value1")
        fact = self.memory_manager.search_facts("key1")
        self.assertEqual(fact, "value1")

    def test_get_conversation_history(self):
        self.memory_manager.store_message("user", "First message")
        self.memory_manager.store_message("assistant", "First response")
        history = self.memory_manager.get_conversation_history()
        self.assertEqual(len(history), 2)

    def test_search_facts(self):
        self.memory_manager.store_fact("key2", "value2")
        result = self.memory_manager.search_facts("key2")
        self.assertEqual(result, "value2")

    def tearDown(self):
        self.memory_manager.close()  # Close the in-memory database

if __name__ == '__main__':
    unittest.main()