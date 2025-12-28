"""OpenAI LLM provider."""

import json
from typing import Optional, Any, Union
from timesketch.lib.llms.providers import interface, manager

# Check if the required dependencies are installed.
has_required_deps = True
try:
    from openai import OpenAI as OpenAIClient
    from openai import OpenAIError, APIError, APITimeoutError, APIConnectionError
except ImportError:
    has_required_deps = False

# Default configuration values
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
                are missing or if the openai package is not installed.
        """
        super().__init__(config, **kwargs)
        
        if not has_required_deps:
            raise ValueError(
                "OpenAI provider requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model")
        timeout = self.config.get("timeout", DEFAULT_TIMEOUT)
        
        if not self.api_key:
            raise ValueError("api_key is required for OpenAI provider")
        if not self.model:
            raise ValueError("model is required for OpenAI provider")
        
        # Initialize OpenAI client
        self.client = OpenAIClient(
            api_key=self.api_key,
            timeout=timeout,
        )

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
        # Prepare request parameters
        create_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            # OpenAI uses 'max_tokens', but we use 'max_output_tokens' for consistency
            # with other LLM providers in Timesketch
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
            create_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**create_kwargs)
            response_data = response.choices[0].message.content
        except APITimeoutError as error:
            raise ValueError(f"Request timed out: {error}") from error
        except (APIConnectionError, APIError) as error:
            raise ValueError(f"Error making request: {error}") from error
        except (AttributeError, IndexError, KeyError) as e:
            # Don't expose full response in error to avoid leaking sensitive data
            raise ValueError(
                "Unexpected response structure from OpenAI API. "
                "Please check your model and API configuration."
            ) from e
        except OpenAIError as error:
            raise ValueError(f"OpenAI API error: {error}") from error

        if response_schema:
            try:
                return json.loads(response_data)
            except json.JSONDecodeError as error:
                # Truncate response data to avoid exposing sensitive content
                truncated = (
                    response_data[:100] + "..."
                    if len(response_data) > 100
                    else response_data
                )
                raise ValueError(
                    f"Error JSON parsing response (first 100 chars): {truncated}"
                ) from error

        return response_data


manager.LLMManager.register_provider(OpenAI)
