import unittest
from src.wrapper import OllamaWrapper
from unittest.mock import patch, MagicMock

class TestStreamChat(unittest.TestCase):

    @patch('src.wrapper.requests.post')
    def test_stream_chat_success(self, mock_post):
        # Mock the response from the requests.post call
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'{"message": "Hello, how can I help you?"}',
            b'{"message": "Here is some information."}'
        ]
        mock_post.return_value = mock_response

        wrapper = OllamaWrapper()
        response = wrapper.stream_chat("Hello")

        # Check that the response is as expected
        self.assertEqual(len(response), 2)
        self.assertEqual(response[0]['message'], "Hello, how can I help you?")
        self.assertEqual(response[1]['message'], "Here is some information.")

    @patch('src.wrapper.requests.post')
    def test_stream_chat_failure(self, mock_post):
        # Mock the response to simulate a failure
        mock_post.side_effect = Exception("Network error")

        wrapper = OllamaWrapper()
        with self.assertRaises(Exception) as context:
            wrapper.stream_chat("Hello")

        self.assertTrue("Network error" in str(context.exception))

    @patch('src.wrapper.requests.post')
    def test_stream_chat_partial_json(self, mock_post):
        # Mock the response with partial JSON
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'{"message": "Hello, how can I help you?"}',
            b'{"message": "This is a partial message',
            b' and should be handled."}'
        ]
        mock_post.return_value = mock_response

        wrapper = OllamaWrapper()
        response = wrapper.stream_chat("Hello")

        # Check that the response handles partial JSON correctly
        self.assertEqual(len(response), 2)
        self.assertEqual(response[0]['message'], "Hello, how can I help you?")
        self.assertEqual(response[1]['message'], "This is a partial message and should be handled.")

if __name__ == '__main__':
    unittest.main()