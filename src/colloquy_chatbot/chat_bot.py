from typing import List, Optional, Callable
from .message import Message, TextMessage
from .prompt_function import catalog_functions


class ChatBot[T]:
    def __init__(self,
                 instructions: Optional[str] = None,
                 functions: Optional[List[Callable]] = None):
        self.instructions = instructions
        self.functions = catalog_functions(functions or [])
        self.history: List[Message] = []
        self.initialize()

    def initialize(self):
        pass

    async def prompt(self, text: str) -> str:
        self.history.append(self.user(text))
        response = await self.send_prompt()
        self.history.append(response)
        return self.output(response)

    async def send_prompt(self) -> Message:
        raise NotImplementedError()

    def output(self, message: Message) -> str:
        if isinstance(message, TextMessage):
            return message.text
        raise NotImplementedError()

    def user(self, text: str) -> Message:
        raise NotImplementedError()
