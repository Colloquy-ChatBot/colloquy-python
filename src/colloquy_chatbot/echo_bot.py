from .message import TextMessage
from .chat_bot import ChatBot


class EchoBot(ChatBot):
    async def send_prompt(self) -> TextMessage:
        last_message = self.history[-1]
        assert isinstance(last_message, TextMessage)
        return TextMessage(last_message.text)

    def user(self, text):
        return TextMessage(text)
