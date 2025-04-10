"""Colloquy Chatbot - A more intuitive interface on top of existing chatbot APIs."""

from .chat_bot import ChatBot, BotMessage
from .claude_bot import ClaudeBot
from .echo_bot import EchoBot
from .openai_bot import OpenAIBot
from .prompt_function import prompt_function, get_llm_functions

__version__ = "0.1.0"
