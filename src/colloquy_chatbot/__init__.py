"""Colloquy Chatbot - A more intuitive interface on top of existing chatbot APIs."""

from .message import TextMessage, RoleMessage
from .chat_bot import ChatBot
from .claude_bot import ClaudeBot
from .echo_bot import EchoBot
from .openai_bot import OpenAIBot
from .prompt_function import get_llm_functions, prompt_function

__version__ = "0.1.0"
