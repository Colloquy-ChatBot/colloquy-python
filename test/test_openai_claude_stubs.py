"""Stub tests for OpenAI and Claude bots.

These tests are meant to be run only with mocked API calls.
They will fail if run against the actual APIs without proper mocking.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from colloquy_chatbot import ClaudeBot, OpenAIBot, prompt_function


@pytest.mark.asyncio
@patch("openai.OpenAI")
async def test_openai_bot_initialization(mock_openai):
    """Test that OpenAIBot can be initialized with proper mocking."""
    # Set up mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Initialize with fake API key
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"}):
        bot = OpenAIBot(instructions="You are a test bot.")

        assert bot.model == "gpt-4o"
        assert bot.instructions == "You are a test bot."
        assert len(bot.history) == 0
        assert bot.api_key == "fake-api-key"


@pytest.mark.asyncio
@patch("anthropic.Anthropic")
async def test_claude_bot_initialization(mock_anthropic):
    """Test that ClaudeBot can be initialized with proper mocking."""
    # Set up mock
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    # Initialize with fake API key
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-api-key"}):
        bot = ClaudeBot(instructions="You are a test bot.")

        assert bot.model == "claude-3-opus-20240229"
        assert bot.instructions == "You are a test bot."
        assert len(bot.history) == 0
        assert bot.api_key == "fake-api-key"


@pytest.mark.asyncio
@patch("openai.OpenAI")
async def test_openai_function_registration(mock_openai):
    """Test that functions are properly registered with OpenAI."""
    # Set up mock client and response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock the completions method
    mock_message = MagicMock()
    mock_message.content = "Function called successfully"
    mock_message.tool_calls = []

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_response

    # Create function and bot
    @prompt_function(description="Test function")
    def test_func(x=1, y=2):
        return x + y

    # Initialize with fake API key
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"}):
        bot = OpenAIBot(functions=[test_func])

        # Send a prompt
        response = await bot.prompt("Test message")

        # Verify the API was called with our function
        tools_arg = mock_client.chat.completions.create.call_args[1]["tools"]
        assert len(tools_arg) == 1
        assert tools_arg[0]["type"] == "function"
        assert tools_arg[0]["function"]["name"] == "test_func"
        assert "parameters" in tools_arg[0]["function"]
