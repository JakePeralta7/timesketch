# Copyright 2025 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for OpenAI LLM provider."""

import json
import unittest
from unittest import mock
from timesketch.lib.llms.providers.contrib.openai import OpenAI


class MockChoice:
    """Mock OpenAI choice."""
    
    def __init__(self, content):
        self.message = mock.Mock()
        self.message.content = content


class MockChatCompletion:
    """Mock OpenAI chat completion response."""
    
    def __init__(self, content):
        self.choices = [MockChoice(content)]


class TestOpenAI(unittest.TestCase):
    """Tests for the OpenAI provider."""

    def test_init_missing_api_key(self):
        """Test that initialization fails without api_key."""
        config = {"model": "gpt-4"}
        with self.assertRaises(ValueError) as context:
            OpenAI(config)
        self.assertIn("api_key is required", str(context.exception))

    def test_init_missing_model(self):
        """Test that initialization fails without model."""
        config = {"api_key": "test-key"}
        with self.assertRaises(ValueError) as context:
            OpenAI(config)
        self.assertIn("model is required", str(context.exception))

    def test_init_success(self):
        """Test successful initialization."""
        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)
        self.assertEqual(provider.api_key, "test-key")
        self.assertEqual(provider.model, "gpt-4")
        self.assertIsNotNone(provider.client)

    def test_init_custom_base_url(self):
        """Test initialization with custom base_url."""
        config = {
            "api_key": "test-key",
            "model": "gpt-4",
            "base_url": "https://custom.openai.com",
        }
        provider = OpenAI(config)
        # Check that the client was initialized with custom base_url
        # OpenAI client normalizes base_url with trailing slash
        self.assertIn("custom.openai.com", str(provider.client.base_url))

    def test_generate_text(self):
        """Test text generation."""
        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)
        
        # Mock the client's chat.completions.create method
        mock_response = MockChatCompletion("This is a test response")
        provider.client.chat.completions.create = mock.Mock(return_value=mock_response)
        
        result = provider.generate("Test prompt")
        
        self.assertEqual(result, "This is a test response")
        provider.client.chat.completions.create.assert_called_once()
        
        # Verify the call parameters
        call_kwargs = provider.client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4")
        self.assertEqual(call_kwargs["messages"], [{"role": "user", "content": "Test prompt"}])

    def test_generate_with_json_schema(self):
        """Test generation with JSON schema."""
        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)
        
        # Mock the client's chat.completions.create method
        mock_response = MockChatCompletion('{"answer": "test answer"}')
        provider.client.chat.completions.create = mock.Mock(return_value=mock_response)
        
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        result = provider.generate("Test prompt", response_schema=schema)
        
        self.assertEqual(result, {"answer": "test answer"})
        
        # Verify that response_format was set
        call_kwargs = provider.client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})

    def test_generate_request_error(self):
        """Test handling of request errors."""
        from openai import APIConnectionError
        
        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)
        
        # Mock the client to raise an error
        provider.client.chat.completions.create = mock.Mock(
            side_effect=APIConnectionError(request=mock.Mock())
        )
        
        with self.assertRaises(ValueError) as context:
            provider.generate("Test prompt")
        self.assertIn("Error making request", str(context.exception))

    def test_generate_timeout_error(self):
        """Test handling of timeout errors."""
        from openai import APITimeoutError
        
        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)
        
        # Mock the client to raise a timeout error
        provider.client.chat.completions.create = mock.Mock(
            side_effect=APITimeoutError(request=mock.Mock())
        )
        
        with self.assertRaises(ValueError) as context:
            provider.generate("Test prompt")
        self.assertIn("Request timed out", str(context.exception))

    def test_generate_invalid_response_structure(self):
        """Test handling of invalid response structure."""
        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)
        
        # Mock the client to return invalid structure
        mock_response = mock.Mock()
        mock_response.choices = []  # Empty choices list
        provider.client.chat.completions.create = mock.Mock(return_value=mock_response)
        
        with self.assertRaises(ValueError) as context:
            provider.generate("Test prompt")
        self.assertIn("Unexpected response structure", str(context.exception))

    def test_generate_json_parse_error(self):
        """Test handling of JSON parsing errors."""
        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)
        
        # Mock the client to return invalid JSON
        mock_response = MockChatCompletion("not valid json")
        provider.client.chat.completions.create = mock.Mock(return_value=mock_response)
        
        schema = {"type": "object"}
        
        with self.assertRaises(ValueError) as context:
            provider.generate("Test prompt", response_schema=schema)
        self.assertIn("Error JSON parsing response", str(context.exception))


if __name__ == "__main__":
    unittest.main()
