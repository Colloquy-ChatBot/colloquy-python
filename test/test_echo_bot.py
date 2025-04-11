"""Tests for the EchoBot class."""

import pytest

from colloquy_chatbot import EchoBot


@pytest.mark.asyncio
async def test_echo_bot():
    """Test that EchoBot echoes back the input message."""
    bot = EchoBot()

    response = await bot.prompt("Hello there!")
    assert response == "Hello there!"

    # Test another message
    response = await bot.prompt("Testing 123")
    assert response == "Testing 123"

    # Check history
    assert len(bot.history) == 4  # 2 prompts + 2 responses
    assert bot.history[0].text == "Hello there!"
    assert bot.history[1].text == "Hello there!"
    assert bot.history[2].text == "Testing 123"
    assert bot.history[3].text == "Testing 123"
