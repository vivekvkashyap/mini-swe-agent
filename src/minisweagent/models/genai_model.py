"""Native Google GenAI model provider."""

import os
from typing import Any

from google import genai
from pydantic import BaseModel, field_validator

from minisweagent.models import GLOBAL_MODEL_STATS


class GenaiProviderOptions(BaseModel):
    """Provider-specific options for Google GenAI."""

    thinking_mode: bool = False
    thinking_budget: int = 10000
    grounding: bool = False
    context_caching: bool = False
    code_execution: bool = False


class GenaiModelConfig(BaseModel):
    """Configuration for GenaiModel."""

    model_name: str
    provider_options: GenaiProviderOptions = GenaiProviderOptions()

    @field_validator("provider_options", mode="before")
    @classmethod
    def parse_provider_options(cls, v):
        """Allow provider_options to be passed as dict."""
        if isinstance(v, dict):
            return GenaiProviderOptions(**v)
        return v


class GenaiModel:
    """Native Google GenAI model using the official client."""

    def __init__(self, *, config_class: type = GenaiModelConfig, **kwargs):
        # Handle nested provider_options dict
        if "provider_options" in kwargs and isinstance(kwargs["provider_options"], dict):
            kwargs["provider_options"] = GenaiProviderOptions(**kwargs["provider_options"])

        self.config = config_class(**kwargs)

        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

        self.cost = 0.0
        self.n_calls = 0

    def _convert_messages(self, messages: list[dict]) -> tuple[list, str | None]:
        """Convert from OpenAI message format to genai Content format."""
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(
                    genai.types.Content(
                        role="user",
                        parts=[genai.types.Part(text=content)],
                    )
                )
            elif role == "assistant":
                contents.append(
                    genai.types.Content(
                        role="model",
                        parts=[genai.types.Part(text=content)],
                    )
                )

        return contents, system_instruction

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the Gemini model."""
        # Merge config defaults with per-call overrides
        options = self.config.provider_options.model_dump()
        options.update(kwargs)

        # Convert messages
        contents, system_instruction = self._convert_messages(messages)

        # Build generation config
        gen_config_kwargs = {"system_instruction": system_instruction}

        # Thinking mode
        if options.get("thinking_mode"):
            gen_config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
                thinking_budget=options.get("thinking_budget", 10000)
            )

        # Build tools list
        tools = []
        if options.get("grounding"):
            tools.append(genai.types.Tool(google_search=genai.types.GoogleSearch()))
        if options.get("code_execution"):
            tools.append(genai.types.Tool(code_execution=genai.types.ToolCodeExecution()))

        if tools:
            gen_config_kwargs["tools"] = tools

        # Make the API call
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            config=genai.types.GenerateContentConfig(**gen_config_kwargs),
        )

        # Track cost
        cost = self._calculate_cost(response.usage_metadata)
        self.cost += cost
        self.n_calls += 1
        GLOBAL_MODEL_STATS.add(cost)

        # Extract response text
        response_text = response.text if response.text else ""

        return {
            "content": response_text,
            "extra": {
                "usage_metadata": {
                    "prompt_token_count": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "candidates_token_count": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_token_count": getattr(response.usage_metadata, "total_token_count", 0),
                },
                "model": self.config.model_name,
            },
        }

    def _calculate_cost(self, usage_metadata) -> float:
        """Calculate cost from Gemini usage metadata."""
        # Gemini 2.5 Pro pricing (per 1M tokens)
        INPUT_COST_PER_1M = 1.25
        OUTPUT_COST_PER_1M = 10.00

        input_tokens = getattr(usage_metadata, "prompt_token_count", 0)
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0)

        return (
            input_tokens * INPUT_COST_PER_1M / 1_000_000
            + output_tokens * OUTPUT_COST_PER_1M / 1_000_000
        )

    def get_template_vars(self) -> dict[str, Any]:
        """Return config for trajectory logging."""
        return self.config.model_dump() | {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }
