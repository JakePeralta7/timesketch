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
"""OpenAI LLM provider."""

import json
from typing import Optional, Any, Union
import requests
from timesketch.lib.llms.providers import interface, manager

# Default configuration values
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 60


class OpenAI(interface.LLMProvider):
    """OpenAI provider for Timesketch.

    This provider uses the OpenAI API to generate text.
    It requires the api_key and model to be configured.
    """

    NAME = "openai"

    def __init__(self, config: dict, **kwargs: Any):
        """Initializes the OpenAI provider.

        Args:
            config (dict): A dictionary of provider-specific configuration options.
            **kwargs (Any): Additional arguments passed to the base class.

        Raises:
            ValueError: If required configuration keys (api_key, model)
                are missing.
        """
        super().__init__(config, **kwargs)
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model")
        self.base_url = self.config.get("base_url", DEFAULT_BASE_URL)
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)
        
        if not self.api_key:
            raise ValueError("api_key is required for OpenAI provider")
        if not self.model:
            raise ValueError("model is required for OpenAI provider")

    def generate(
        self, prompt: str, response_schema: Optional[dict] = None
    ) -> Union[dict, str]:
        """Generate text using the OpenAI API.

        Args:
            prompt: The prompt to use for generation.
            response_schema: An optional JSON schema to define the expected
                response format.

        Returns:
            The generated text as a string, or parsed JSON if response_schema
            is provided.

        Raises:
            ValueError: If the request fails or JSON parsing fails.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.get(
                "max_output_tokens", interface.DEFAULT_MAX_OUTPUT_TOKENS
            ),
            "temperature": self.config.get(
                "temperature", interface.DEFAULT_TEMPERATURE
            ),
            "top_p": self.config.get("top_p", interface.DEFAULT_TOP_P),
        }

        # Add response format for structured output if schema is provided
        if response_schema:
            data["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(
                url, headers=headers, json=data, timeout=self.timeout
            )
            response.raise_for_status()
            response_json = response.json()
            response_data = response_json["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout as error:
            raise ValueError(f"Request timed out: {error}") from error
        except requests.exceptions.RequestException as error:
            raise ValueError(f"Error making request: {error}") from error
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected response structure from OpenAI API: {response_json}"
            ) from e

        if response_schema:
            try:
                return json.loads(response_data)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Error JSON parsing text: {response_data}: {error}"
                ) from error

        return response_data


manager.LLMManager.register_provider(OpenAI)
