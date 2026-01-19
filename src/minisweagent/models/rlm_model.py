"""RLM Model wrapper implementing the mini-swe-agent Model protocol."""

import os
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from minisweagent.models import GLOBAL_MODEL_STATS


class RLMModelConfig(BaseModel):
    model_name: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


class RLMModel:
    """RLM Model wrapper that implements the mini-swe-agent Model protocol."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        model_kwargs: dict | None = None,
        **kwargs,
    ):
        model_kwargs = model_kwargs or {}
        self.config = RLMModelConfig(
            model_name=model_name,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            temperature=model_kwargs.get("temperature", 0.0),
            max_tokens=model_kwargs.get("max_tokens"),
        )

        if not self.config.api_key:
            msg = "OPENAI_API_KEY environment variable is required"
            raise ValueError(msg)

        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

        self.cost: float = 0.0
        self.n_calls: int = 0
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the model and return the response."""
        # Filter messages to only include role and content
        clean_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=clean_messages,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
                **kwargs,
            )

            self.n_calls += 1

            # Track token usage and estimate cost
            if response.usage:
                self._input_tokens += response.usage.prompt_tokens
                self._output_tokens += response.usage.completion_tokens
                # Rough cost estimate (varies by model)
                self.cost += (response.usage.prompt_tokens * 0.003 + response.usage.completion_tokens * 0.015) / 1000

            GLOBAL_MODEL_STATS.add(self.cost)

            content = response.choices[0].message.content or ""
            return {"content": content}

        except Exception as e:
            msg = f"Error generating completion: {e!s}"
            raise RuntimeError(msg) from e

    def get_template_vars(self) -> dict[str, Any]:
        """Return template variables for Jinja rendering."""
        return {
            "model_name": self.config.model_name,
            "cost": self.cost,
            "n_calls": self.n_calls,
        }

