import unittest
from src.wrapper import OllamaWrapper
from src.memory_manager import MemoryManager

class TestBuildMessages(unittest.TestCase):

    def setUp(self):
        self.wrapper = OllamaWrapper()
        self.memory_manager = MemoryManager()

    def test_build_messages_with_system_prompt(self):
        system_prompt = "You are a helpful assistant."
        user_message = "What is the capital of France?"
        self.memory_manager.store_message("user", user_message)
        expected_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        built_messages = self.wrapper._build_messages(system_prompt=system_prompt)
        self.assertEqual(built_messages, expected_messages)

    def test_build_messages_without_system_prompt(self):
        user_message = "What is the capital of Germany?"
        self.memory_manager.store_message("user", user_message)
        expected_messages = [
            {"role": "user", "content": user_message}
        ]
        built_messages = self.wrapper._build_messages()
        self.assertEqual(built_messages, expected_messages)

    def test_build_messages_with_previous_history(self):
        system_prompt = "You are a helpful assistant."
        user_message_1 = "What is the capital of Italy?"
        assistant_response_1 = "The capital of Italy is Rome."
        user_message_2 = "What about Spain?"
        assistant_response_2 = "The capital of Spain is Madrid."

        self.memory_manager.store_message("user", user_message_1)
        self.memory_manager.store_message("assistant", assistant_response_1)
        self.memory_manager.store_message("user", user_message_2)
        self.memory_manager.store_message("assistant", assistant_response_2)

        expected_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_1},
            {"role": "assistant", "content": assistant_response_1},
            {"role": "user", "content": user_message_2},
            {"role": "assistant", "content": assistant_response_2}
        ]
        built_messages = self.wrapper._build_messages(system_prompt=system_prompt)
        self.assertEqual(built_messages, expected_messages)

if __name__ == '__main__':
    unittest.main()