import os
from typing import Any, Callable, Dict, List, Optional, cast

import openai
from openai.types.chat import ChatCompletion

from .message import TextMessage
from .openai.message import RoleMessage, FunctionCallMessage
from .openai.function import get_llm_metadata
from .chat_bot import ChatBot
from .prompt_function import get_llm_functions


class OpenAIBot(ChatBot):
    def initialize(self):
        self.client = openai.AsyncOpenAI()

        if self.instructions:
            self.history.append(RoleMessage("system", self.instructions))

    async def send_prompt(self) -> RoleMessage:
        response = await self.client.responses.create(
            model="gpt-4o-mini",
            input=[message.input() for message in self.history],
            tools=[get_llm_metadata(func) for func in self.functions.values()],
        )
        print(response)

        response_content = ""
        for content in response.output:
            if content.type == "function_call":
                message = FunctionCallMessage(self.functions[content.name], content)
                self.history.append(message)
                self.history.append(message.invoke())
            elif content.type == "message":
                return RoleMessage("assistant", content.content[0].text)
            else:
                raise NotImplementedError("Unexpected content type", content)

        return await self.send_prompt()

    def user(self, text):
        return RoleMessage("user", text)
