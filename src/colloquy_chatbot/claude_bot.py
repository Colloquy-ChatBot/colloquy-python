"""Claude implementation of the ChatBot."""

import os
import json
from typing import List, Optional, Dict, Any, Callable

import anthropic

from .chat_bot import BotMessage, ChatBot
from .prompt_function import get_llm_functions


class ClaudeBot(ChatBot):
    """ChatBot implementation using Anthropic's Claude API."""

    def __init__(
        self,
        instructions: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        functions: Optional[List[Callable]] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize a Claude chatbot.

        Args:
            instructions: Optional system instructions for the bot
            model: The Claude model to use
            functions: Optional list of functions available to the bot
            api_key: Optional API key (defaults to ANTHROPIC_API_KEY env var)
        """
        super().__init__(instructions)
        self.model = model
        self.functions = functions or []
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)

    async def send_prompt(self) -> BotMessage:
        """Send the current conversation to Claude and get a response.

        Returns:
            A BotMessage containing the model's response
        """
        messages = []
        
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
                    tools.append({
                        "type": "function",
                        "function": function_data
                    })
        
        system = self.instructions if self.instructions else "You are a helpful assistant."
        
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            tools=tools,
        )
        
        content = response.content[0].text
        
        # Handle tool calls
        tool_calls = response.tool_use or []
        for tool_call in tool_calls:
            if tool_call.type == "function":
                function_name = tool_call.name
                function_args = tool_call.input
                
                # Find the matching function
                for func in self.functions:
                    if getattr(func, "__llm_metadata__", {}).get("name") == function_name:
                        # Execute the function
                        if isinstance(function_args, str):
                            function_args = json.loads(function_args)
                        result = func(**function_args)
                        content += f"\n\nFunction result: {result}"
                        break
        
        return BotMessage(content)