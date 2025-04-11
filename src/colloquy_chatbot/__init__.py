"""Colloquy Chatbot - A more intuitive interface on top of existing chatbot APIs."""

from .chat_bot import BotMessage, ChatBot
from .claude_bot import ClaudeBot
from .echo_bot import EchoBot
from .openai_bot import OpenAIBot
from .prompt_function import get_llm_functions, prompt_function

__version__ = "0.1.0"
