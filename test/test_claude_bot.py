import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from unittest.mock import patch, AsyncMock, call

from colloquy_chatbot import ClaudeBot, prompt_function
from colloquy_chatbot.claude.message import RoleMessage, FunctionCallMessage, FunctionResultMessage

usage = Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=388, output_tokens=55)

# Define a function for the bot to use
@prompt_function(description="Get current weather for a location")
def get_weather(location: str, unit: str = "celsius"):
    # In a real app, this would call a weather API
    return f"Weather in {location}: 22°{'C' if unit == 'celsius' else 'F'}, Sunny"

def message(content):
    return Message(
        id='msg_01VQye61h5bwgLUH8VUf6iWA',
        content=content,
        model='claude-3-7-sonnet-20250219',
        role='assistant',
        type='message',
        usage=usage
    )

@pytest.mark.asyncio
@patch("colloquy_chatbot.claude_bot.anthropic.AsyncAnthropic")
async def test_simple_prompt(mock_anthropic):
    mock = AsyncMock()
    mock_anthropic.return_value = mock

    mock.messages.create.side_effect = [message([
        TextBlock(
            type='text',
            text="Hi",
        ),
    ])]

    bot = ClaudeBot()
    response = await bot.prompt("Hello")
    assert response == "Hi"

@pytest.mark.asyncio
@patch("colloquy_chatbot.claude_bot.anthropic.AsyncAnthropic")
async def test_functions(mock_anthropic):
    mock = AsyncMock()
    mock_anthropic.return_value = mock

    # Create bot with access to this function
    bot = ClaudeBot(
        functions=[get_weather],
    )

    assistant_looking_up_text = "Alright, looking up the weather in Tokyo"
    assistant_result_text = "The weather is 22 degrees celsius"
    tool_use = ToolUseBlock(
        type='tool_use',
        id='toolu_01CWuduGjgSRfsYJUNC2wxi7',
        input={'location': 'Tokyo', 'unit': 'celsius'},
        name='get_weather',
    )
    mock.messages.create.side_effect = [
        message([
            TextBlock(
                type='text',
                text=assistant_looking_up_text,
            ),
            tool_use,
        ]),
        message([
            TextBlock(
                type='text',
                text=assistant_result_text,
            ),
        ]),
    ]

    user_text = "What's the weather in Tokyo?"
    response = await bot.prompt(user_text)

    def build_response(messages):
        return call(
            model = "claude-3-7-sonnet-20250219",
            max_tokens = 1000,
            messages = messages,
            tools = [{
                "name": "get_weather",
                "description": "Get current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "description": "",
                            "type": "string",
                        },
                        "unit": {
                            "description": "",
                            "type": "string",
                        }
                    }
                }
            }]
        )

    mock.messages.create.assert_has_calls([
        build_response([{"role": "user", "content": "What's the weather in Tokyo?"}]),
        build_response([
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "content": assistant_looking_up_text},
            {"role": "assistant", "content": [tool_use]},
            {
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": get_weather(**tool_use.input),
                }],
                "role": "user",
            },
        ])
    ])

    assert bot.history == [
        RoleMessage("user", user_text),
        RoleMessage("assistant", assistant_looking_up_text),
        FunctionCallMessage(get_weather, tool_use),
        FunctionResultMessage(tool_use.id, get_weather(**tool_use.input)),
        RoleMessage("assistant", assistant_result_text),
    ]
