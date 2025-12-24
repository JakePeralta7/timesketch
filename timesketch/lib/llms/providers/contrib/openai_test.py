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


class MockResponse:
    """Mock HTTP response."""

    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception(f"HTTP {self.status_code}")


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
        self.assertEqual(provider.base_url, "https://api.openai.com/v1")

    def test_init_custom_base_url(self):
        """Test initialization with custom base_url."""
        config = {
            "api_key": "test-key",
            "model": "gpt-4",
            "base_url": "https://custom.openai.com",
        }
        provider = OpenAI(config)
        self.assertEqual(provider.base_url, "https://custom.openai.com")

    @mock.patch("timesketch.lib.llms.providers.contrib.openai.requests.post")
    def test_generate_text(self, mock_post):
        """Test text generation."""
        mock_response = MockResponse(
            {
                "choices": [
                    {"message": {"content": "This is a test response"}}
                ]
            },
            200,
        )
        mock_post.return_value = mock_response

        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)
        result = provider.generate("Test prompt")

        self.assertEqual(result, "This is a test response")
        mock_post.assert_called_once()
        
        # Verify the request parameters
        call_args = mock_post.call_args
        self.assertIn("https://api.openai.com/v1/chat/completions", call_args[0])
        self.assertEqual(call_args[1]["headers"]["Authorization"], "Bearer test-key")

    @mock.patch("timesketch.lib.llms.providers.contrib.openai.requests.post")
    def test_generate_with_json_schema(self, mock_post):
        """Test generation with JSON schema."""
        mock_response = MockResponse(
            {
                "choices": [
                    {"message": {"content": '{"answer": "test answer"}'}}
                ]
            },
            200,
        )
        mock_post.return_value = mock_response

        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        result = provider.generate("Test prompt", response_schema=schema)

        self.assertEqual(result, {"answer": "test answer"})
        
        # Verify that response_format was set
        call_args = mock_post.call_args
        self.assertEqual(
            call_args[1]["json"]["response_format"], {"type": "json_object"}
        )

    @mock.patch("timesketch.lib.llms.providers.contrib.openai.requests.post")
    def test_generate_request_error(self, mock_post):
        """Test handling of request errors."""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)

        with self.assertRaises(ValueError) as context:
            provider.generate("Test prompt")
        self.assertIn("Error making request", str(context.exception))

    @mock.patch("timesketch.lib.llms.providers.contrib.openai.requests.post")
    def test_generate_invalid_response_structure(self, mock_post):
        """Test handling of invalid response structure."""
        mock_response = MockResponse({"invalid": "structure"}, 200)
        mock_post.return_value = mock_response

        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)

        with self.assertRaises(ValueError) as context:
            provider.generate("Test prompt")
        self.assertIn("Unexpected response structure", str(context.exception))

    @mock.patch("timesketch.lib.llms.providers.contrib.openai.requests.post")
    def test_generate_json_parse_error(self, mock_post):
        """Test handling of JSON parsing errors."""
        mock_response = MockResponse(
            {
                "choices": [
                    {"message": {"content": "not valid json"}}
                ]
            },
            200,
        )
        mock_post.return_value = mock_response

        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAI(config)
        schema = {"type": "object"}

        with self.assertRaises(ValueError) as context:
            provider.generate("Test prompt", response_schema=schema)
        self.assertIn("Error JSON parsing response", str(context.exception))


if __name__ == "__main__":
    unittest.main()
