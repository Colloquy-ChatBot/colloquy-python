"""OpenAI implementation of the ChatBot."""

import os
from typing import Any, Callable, Dict, List, Optional, cast

import openai
from openai.types.chat import ChatCompletion

from .chat_bot import BotMessage, ChatBot
from .prompt_function import get_llm_functions


class OpenAIBot(ChatBot):
    """ChatBot implementation using OpenAI's API."""

    def __init__(
        self,
        instructions: Optional[str] = None,
        model: str = "gpt-4o",
        functions: Optional[List[Callable]] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize an OpenAI chatbot.

        Args:
            instructions: Optional system instructions for the bot
            model: The OpenAI model to use
            functions: Optional list of functions available to the bot
            api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        """
        super().__init__(instructions)
        self.model = model
        self.functions = functions or []
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key."
            )

        self.client = openai.OpenAI(api_key=self.api_key)

    async def send_prompt(self) -> BotMessage:
        """Send the current conversation to OpenAI and get a response.

        Returns:
            A BotMessage containing the model's response
        """
        messages = []

        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})

        for i, message in enumerate(self.history):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": message.text})

        # Set up function calling if functions are provided
        tools = None
        if self.functions:
            tools = []
            for func in self.functions:
                function_data = getattr(func, "__llm_metadata__", None)
                if function_data:
                    tools.append({"type": "function", "function": function_data})

        # Convert messages to format expected by OpenAI
        api_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            api_messages.append({"role": role, "content": content})

        # Create chat completion
        if tools:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,  # type: ignore
                tools=tools,  # type: ignore
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,  # type: ignore
            )

        response = cast(ChatCompletion, response)

        content = response.choices[0].message.content or ""

        # Handle tool calls
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.type == "function":
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments

                    # Find the matching function
                    for func in self.functions:
                        if (
                            getattr(func, "__llm_metadata__", {}).get("name")
                            == function_name
                        ):
                            # Execute the function
                            import json

                            result = func(**json.loads(function_args))
                            content += f"\n\nFunction result: {result}"
                            break

        return BotMessage(content)
