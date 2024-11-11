import json
import os
import random
from anthropic import Anthropic
from anthropic.types import ToolUseBlock, ToolResultBlockParam

from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-3-5-sonnet-20241022"

SYS_PROMPT = """
You are a friendly chatbot answering the user's questions.
"""

user_msg = "What's the weather like in Helsinki?"


tools = [
    {
        "name": "getWeather",
        "description": "Get weather information about a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the city",
                }
            },
            "required": ["city"],
        },
    }
]


def handle_tool_call(tool_call: ToolUseBlock):
    tool_name = tool_call.name

    print(f"LLM called tool: {tool_name} with args: {tool_call.input}")

    temperature = random.randint(0, 20)
    weather_type = random.choice(["sunny", "cloudy", "raining"])

    if tool_name == "getWeather":
        city = tool_call.input["city"]

        return ToolResultBlockParam(
            type="tool_result",
            tool_use_id=tool_call.id,
            content=f"The weather in {city} today: {temperature}°, {weather_type}",
        )
    else:
        raise ValueError(f"Unknown tool {tool_name}")


def call_llm(messages, tools=None):
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=messages,
        system=SYS_PROMPT,
        tools=tools,
    )
    return response


def stream_llm(messages, tools=None):
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        messages=messages,
        system=SYS_PROMPT,
        tools=tools,
    ) as stream:
        answer = ""
        for text in stream.text_stream:
            answer += text or ""
            print(text, end="", flush=True)

        return answer


messages = [{"role": "user", "content": user_msg}]

response = call_llm(messages, tools=tools)

if response.stop_reason == "end_turn":
    print(response.content[0].text)
elif response.stop_reason == "tool_use":
    messages.append({"role": "assistant", "content": response.content})

    for block in response.content:
        if block.type == "text":
            print(block.text)
        elif block.type == "tool_use":
            tool_result = handle_tool_call(block)
            messages.append({"role": "user", "content": [tool_result]})

    stream_llm(messages=messages, tools=tools)
