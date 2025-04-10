"""Base ChatBot class that all bots inherit from."""

from typing import List, Optional


class BotMessage:
    """Representation of a message from a bot."""

    def __init__(self, text: str):
        """Initialize a new bot message.

        Args:
            text: The message text
        """
        self.text = text

    def __str__(self) -> str:
        return self.text


class ChatBot:
    """Base class for all chatbots."""

    def __init__(self, instructions: Optional[str] = None):
        """Initialize a new chatbot.

        Args:
            instructions: Optional system instructions for the bot
        """
        self.instructions = instructions
        self.history: List[BotMessage] = []

    async def send_prompt(self) -> BotMessage:
        """Send the current prompt to the chatbot and get a response.

        This method should be implemented by subclasses.

        Returns:
            A BotMessage containing the bot's response
        """
        raise NotImplementedError("Subclasses must implement send_prompt")

    async def prompt(self, text: str) -> str:
        """Send a prompt to the chatbot and get a response.

        Args:
            text: The prompt text

        Returns:
            The bot's response as a string
        """
        self.history.append(BotMessage(text))
        response = await self.send_prompt()
        self.history.append(response)
        return response.text
