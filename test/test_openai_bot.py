from unittest.mock import AsyncMock, patch

import pytest

from openai.types.responses.response import Response
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from colloquy_chatbot.openai_bot import OpenAIBot
from colloquy_chatbot.openai.message import RoleMessage, FunctionCallMessage, FunctionResultMessage
from colloquy_chatbot.prompt_function import prompt_function


function_call = ResponseFunctionToolCall(
    arguments='{ "a": "a", "b": 1 }',
    call_id="function_call_id",
    name="concatenate",
    type="function_call",
)

@prompt_function()
def concatenate(a: str, b: int):
    return a + str(b)

def response(output):
    return Response(
        object='response',
        id='id',
        output=output,
        created_at=0,
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata={},
        model='gpt-4o-mini-2024-07-18',
        parallel_tool_calls=True,
        temperature=1.0,
        tool_choice='auto',
        tools=[],
        top_p=1.0,
        max_output_tokens=None,
        previous_response_id=None,
        status='completed',
        truncation='disabled',
        user=None,
        service_tier='default',
        store=True,
    )

@patch("openai.AsyncOpenAI")
def bot(openai, response_text="", instructions=None, functions=False):
    mock = AsyncMock()
    openai.return_value = mock

    responses = []

    if functions:
        responses.append(response([function_call]))

    responses.append(response([ResponseOutputMessage(
        id='id',
        content=[ResponseOutputText(
            annotations=[],
            text=response_text,
            type='output_text'
        )],
        role='assistant',
        status='completed',
        type='message'
    )]))

    if functions:
        responses.append(None)

    mock.responses.create.side_effect = responses

    return OpenAIBot(instructions=instructions, functions=[concatenate] if functions else None)

@pytest.mark.asyncio
async def test_simple_response():
    b = bot(response_text="Hi")
    response = await b.prompt("Hello")
    assert response == "Hi"
    assert b.client.responses.create.call_args[-1] == {
        "model": "gpt-4o-mini",
        "input": [{
            "role": "user",
            "content": "Hello",
        }],
        "tools": [],
    }

@pytest.mark.asyncio
async def test_instructions():
    b = bot(instructions="test")
    await b.prompt("Hello")
    assert b.history[0] == RoleMessage("system", "test")

@pytest.mark.asyncio
async def test_functions():
    b = bot(response_text="Hi", functions=[concatenate])
    await b.prompt("Hello")

    assert len(b.client.responses.create.call_args_list) == 2

    assert [
        {
            'name': 'concatenate',
            'type': 'function',
            'description': '',
            'parameters': {
                "type": "object",
                "properties": {
                    'a': {
                        'description': "",
                        'type': 'string',
                    },
                    'b': {
                        'description': "",
                        'type': 'integer',
                    },
                },
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        }
    ] == b.client.responses.create.call_args_list[0][1]['tools']

    assert [
        {
            "content": "Hello",
            "role": "user",
        },
        function_call,
        {
            "type": "function_call_output",
            "call_id": function_call.call_id,
            "output": concatenate("a", 1),
        },
    ] == b.client.responses.create.call_args_list[1][1]['input']

    function_call_message = FunctionCallMessage(concatenate, function_call)
    assert b.history == [
        RoleMessage("user", "Hello"),
        function_call_message,
        FunctionResultMessage(function_call.call_id, concatenate("a", 1)),
        RoleMessage("assistant", "Hi"),
    ]
