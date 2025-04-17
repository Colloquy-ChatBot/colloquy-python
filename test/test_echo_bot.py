"""Tests for the EchoBot class."""

import pytest

from colloquy_chatbot import EchoBot, RoleMessage, TextMessage


@pytest.mark.asyncio
async def test_prompt():
    bot = EchoBot()
    response = await bot.prompt("Hello there!")
    assert response == "Hello there!"

@pytest.mark.asyncio
async def test_history():
    bot = EchoBot()
    await bot.prompt("Hello there!")
    assert bot.history == [
        TextMessage("Hello there!"),
        TextMessage("Hello there!"),
    ]
