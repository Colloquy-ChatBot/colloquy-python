"""EchoBot implementation for testing."""

from .chat_bot import BotMessage, ChatBot


class EchoBot(ChatBot):
    """A simple echo bot that repeats the last message back."""

    async def send_prompt(self) -> BotMessage:
        """Echo back the last message.

        Returns:
            A BotMessage containing the echoed text
        """
        last_message = self.history[-1]
        return BotMessage(last_message.text)
