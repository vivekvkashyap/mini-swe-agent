import os
from unittest.mock import patch, Mock

import pytest

from minisweagent.models.genai_model import GenaiModelConfig, GenaiProviderOptions


class TestGenaiProviderOptions:
    def test_default_values(self):
        """Test that provider options have correct defaults."""
        options = GenaiProviderOptions()
        assert options.thinking_mode is False
        assert options.thinking_budget == 10000
        assert options.grounding is False
        assert options.context_caching is False
        assert options.code_execution is False

    def test_custom_values(self):
        """Test that provider options accept custom values."""
        options = GenaiProviderOptions(
            thinking_mode=True,
            thinking_budget=20000,
            grounding=True,
        )
        assert options.thinking_mode is True
        assert options.thinking_budget == 20000
        assert options.grounding is True


class TestGenaiModelConfig:
    def test_model_name_required(self):
        """Test that model_name is required."""
        with pytest.raises(Exception):
            GenaiModelConfig()

    def test_model_name_and_defaults(self):
        """Test config with model name and default provider options."""
        config = GenaiModelConfig(model_name="gemini-2.5-pro")
        assert config.model_name == "gemini-2.5-pro"
        assert config.provider_options.thinking_mode is False

    def test_nested_provider_options(self):
        """Test config with nested provider options dict."""
        config = GenaiModelConfig(
            model_name="gemini-2.5-pro",
            provider_options={"thinking_mode": True, "thinking_budget": 5000},
        )
        assert config.provider_options.thinking_mode is True
        assert config.provider_options.thinking_budget == 5000


class TestGenaiModelInit:
    def test_init_with_model_name(self):
        """Test GenaiModel initializes with model name."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client:
                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(model_name="gemini-2.5-pro")

                assert model.config.model_name == "gemini-2.5-pro"
                assert model.cost == 0.0
                assert model.n_calls == 0
                mock_client.assert_called_once()

    def test_init_with_provider_options(self):
        """Test GenaiModel initializes with provider options."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client"):
                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(
                    model_name="gemini-2.5-pro",
                    provider_options={"thinking_mode": True},
                )

                assert model.config.provider_options.thinking_mode is True

    def test_init_uses_gemini_api_key_fallback(self):
        """Test GenaiModel falls back to GEMINI_API_KEY."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}, clear=True):
            with patch("google.genai.Client") as mock_client:
                from minisweagent.models.genai_model import GenaiModel

                GenaiModel(model_name="gemini-2.5-pro")

                mock_client.assert_called_once_with(api_key="gemini-key")


class TestGenaiMessageConversion:
    def test_convert_user_message(self):
        """Test converting user message to genai format."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client"):
                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(model_name="gemini-2.5-pro")
                messages = [{"role": "user", "content": "Hello"}]

                contents, system_instruction = model._convert_messages(messages)

                assert len(contents) == 1
                assert contents[0].role == "user"
                assert contents[0].parts[0].text == "Hello"
                assert system_instruction is None

    def test_convert_system_message(self):
        """Test that system message becomes system_instruction."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client"):
                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(model_name="gemini-2.5-pro")
                messages = [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ]

                contents, system_instruction = model._convert_messages(messages)

                assert len(contents) == 1
                assert system_instruction == "You are helpful"

    def test_convert_assistant_to_model_role(self):
        """Test that assistant role becomes model role."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client"):
                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(model_name="gemini-2.5-pro")
                messages = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]

                contents, system_instruction = model._convert_messages(messages)

                assert len(contents) == 2
                assert contents[0].role == "user"
                assert contents[1].role == "model"
                assert contents[1].parts[0].text == "Hi there"


class TestGenaiModelQuery:
    def test_basic_query(self):
        """Test basic query returns content and extra."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                # Mock response
                mock_response = Mock()
                mock_response.text = "Hello, world!"
                mock_response.usage_metadata = Mock(
                    prompt_token_count=10,
                    candidates_token_count=5,
                    total_token_count=15,
                )
                mock_client.models.generate_content.return_value = mock_response

                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(model_name="gemini-2.5-pro")
                result = model.query([{"role": "user", "content": "Hi"}])

                assert result["content"] == "Hello, world!"
                assert "extra" in result
                assert model.n_calls == 1

    def test_query_tracks_cost(self):
        """Test that query tracks cost from usage metadata."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                mock_response = Mock()
                mock_response.text = "Response"
                mock_response.usage_metadata = Mock(
                    prompt_token_count=1000,
                    candidates_token_count=100,
                    total_token_count=1100,
                )
                mock_client.models.generate_content.return_value = mock_response

                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(model_name="gemini-2.5-pro")
                model.query([{"role": "user", "content": "Hi"}])

                # Cost = (1000 * 1.25 / 1M) + (100 * 10.00 / 1M) = 0.00125 + 0.001 = 0.00225
                assert model.cost > 0
                assert model.n_calls == 1


class TestGenaiModelThinkingMode:
    def test_thinking_mode_from_config(self):
        """Test that thinking mode is configured from provider_options."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                mock_response = Mock()
                mock_response.text = "Response"
                mock_response.usage_metadata = Mock(
                    prompt_token_count=10, candidates_token_count=5, total_token_count=15
                )
                mock_client.models.generate_content.return_value = mock_response

                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(
                    model_name="gemini-2.5-pro",
                    provider_options={"thinking_mode": True, "thinking_budget": 5000},
                )
                model.query([{"role": "user", "content": "Think about this"}])

                # Verify generate_content was called with thinking config
                call_kwargs = mock_client.models.generate_content.call_args
                config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
                assert config.thinking_config is not None
                assert config.thinking_config.thinking_budget == 5000

    def test_thinking_mode_per_call_override(self):
        """Test that thinking mode can be enabled per-call."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                mock_response = Mock()
                mock_response.text = "Response"
                mock_response.usage_metadata = Mock(
                    prompt_token_count=10, candidates_token_count=5, total_token_count=15
                )
                mock_client.models.generate_content.return_value = mock_response

                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(model_name="gemini-2.5-pro")
                model.query(
                    [{"role": "user", "content": "Think"}],
                    thinking_mode=True,
                    thinking_budget=8000,
                )

                call_kwargs = mock_client.models.generate_content.call_args
                config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
                assert config.thinking_config is not None
                assert config.thinking_config.thinking_budget == 8000


class TestGenaiModelTools:
    def test_grounding_enabled(self):
        """Test that grounding adds GoogleSearch tool."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                mock_response = Mock()
                mock_response.text = "Response"
                mock_response.usage_metadata = Mock(
                    prompt_token_count=10, candidates_token_count=5, total_token_count=15
                )
                mock_client.models.generate_content.return_value = mock_response

                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(
                    model_name="gemini-2.5-pro",
                    provider_options={"grounding": True},
                )
                model.query([{"role": "user", "content": "Search something"}])

                call_kwargs = mock_client.models.generate_content.call_args
                config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
                assert config.tools is not None
                assert len(config.tools) == 1

    def test_code_execution_enabled(self):
        """Test that code_execution adds CodeExecution tool."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                mock_response = Mock()
                mock_response.text = "Response"
                mock_response.usage_metadata = Mock(
                    prompt_token_count=10, candidates_token_count=5, total_token_count=15
                )
                mock_client.models.generate_content.return_value = mock_response

                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(
                    model_name="gemini-2.5-pro",
                    provider_options={"code_execution": True},
                )
                model.query([{"role": "user", "content": "Run code"}])

                call_kwargs = mock_client.models.generate_content.call_args
                config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
                assert config.tools is not None
                assert len(config.tools) == 1

    def test_multiple_tools_enabled(self):
        """Test that multiple tools can be enabled."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                mock_response = Mock()
                mock_response.text = "Response"
                mock_response.usage_metadata = Mock(
                    prompt_token_count=10, candidates_token_count=5, total_token_count=15
                )
                mock_client.models.generate_content.return_value = mock_response

                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(model_name="gemini-2.5-pro")
                model.query(
                    [{"role": "user", "content": "Do stuff"}],
                    grounding=True,
                    code_execution=True,
                )

                call_kwargs = mock_client.models.generate_content.call_args
                config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
                assert config.tools is not None
                assert len(config.tools) == 2


class TestGenaiModelTemplateVars:
    def test_get_template_vars(self):
        """Test that get_template_vars returns config and stats."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("google.genai.Client"):
                from minisweagent.models.genai_model import GenaiModel

                model = GenaiModel(
                    model_name="gemini-2.5-pro",
                    provider_options={"thinking_mode": True},
                )
                model.cost = 0.05
                model.n_calls = 3

                vars = model.get_template_vars()

                assert vars["model_name"] == "gemini-2.5-pro"
                assert vars["provider_options"]["thinking_mode"] is True
                assert vars["n_model_calls"] == 3
                assert vars["model_cost"] == 0.05