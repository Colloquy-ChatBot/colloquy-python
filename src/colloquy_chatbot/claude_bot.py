import json
import os
from typing import Any, Callable, Dict, List, Optional, cast

import anthropic
from anthropic.types import MessageParam, TextBlock, MessageCreateParams, ToolUseBlock

from .claude.message import RoleMessage, FunctionCallMessage
from .claude.function import get_llm_metadata
from .chat_bot import ChatBot
from .prompt_function import get_llm_functions, catalog_functions


class ClaudeBot(ChatBot):
    def initialize(self):
        self.client = anthropic.AsyncAnthropic()

    async def send_prompt(self):
        response = await self.client.messages.create(**self.args())

        needs_reply = False
        for content in response.content:
            if content.type == "tool_use":
                self.call_function(content)
                needs_reply = True
            elif content.type == "text":
                self.history.append(RoleMessage("assistant", content.text))

        if needs_reply:
            return await self.send_prompt()
        else:
            return self.history.pop()

    def call_function(self, content: ToolUseBlock):
        call = FunctionCallMessage(
            self.functions[content.name],
            content
        )
        self.history.append(call)
        self.history.append(call.invoke())

    def user(self, text):
        return RoleMessage("user", text)

    def args(self):
        args = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 1000,
            "messages": [message.input() for message in self.history],
            "tools": [
                get_llm_metadata(func)
                for func in self.functions.values()
                if get_llm_metadata(func)
            ],
        }

        if self.instructions:
            args["system"] = self.instructions

        return args
