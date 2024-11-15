import json
import os
import random
from typing import List
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function as ToolCallFunction,
)
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYS_PROMPT = """
You are a friendly chatbot answering the user's questions.
"""

tools: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "getWeather",
            "description": "Get weather information about a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getTraffic",
            "description": "Get traffic information about a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city",
                    }
                },
                "required": ["city"],
            },
        },
    },
]


def handle_tool_call(tool_call: ChatCompletionMessageToolCall):
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    print(f"LLM called tool: {tool_name} with args: {tool_args}")

    if tool_name == "getWeather":
        temperature = random.randint(0, 20)
        weather_type = random.choice(["sunny", "cloudy", "raining"])
        city = tool_args["city"]

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": "getWeather",
            "content": f"The weather in {city} today: {temperature}°, {weather_type}",
        }
    elif tool_name == "getTraffic":
        city = tool_args["city"]
        traffic = random.choice(["light", "moderate", "heavy"])

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": "getWeather",
            "content": f"The traffic in {city} today: {traffic}",
        }
    else:
        raise ValueError(f"Unknown tool {tool_name}")


def call_llm(messages, stream=False, tools=None):
    response = llm.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=messages,
        stream=stream,
        tools=tools,
    )

    return response


user_msg = "what's the weather and traffic like in Helsinki?"

messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": user_msg},
]

stream = call_llm(messages, stream=True, tools=tools)
tool_calls = []

# Parse the possible tool calls coming from the stream.
for chunk in stream:
    if len(chunk.choices) > 0 and chunk.choices[0].delta:
        delta = chunk.choices[0].delta

        if delta.tool_calls:
            for tool_chunk in delta.tool_calls:
                if len(tool_calls) <= tool_chunk.index:
                    tool_calls.append(
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )

                tc = tool_calls[tool_chunk.index]

                if tool_chunk.id:
                    tc["id"] += tool_chunk.id
                if tool_chunk.function.name:
                    tc["function"]["name"] += tool_chunk.function.name
                if tool_chunk.function.arguments:
                    tc["function"][
                        "arguments"
                    ] += tool_chunk.function.arguments
        else:
            # No tool calls, just stream the message
            content = delta.content or ""
            print(content, end="", flush=True)


if len(tool_calls) > 0:
    for tool_call in tool_calls:
        messages.append({"role": "assistant", "tool_calls": [tool_call]})
        tool_response = handle_tool_call(
            tool_call=ChatCompletionMessageToolCall(
                id=tool_call["id"],
                type="function",
                function=ToolCallFunction(
                    name=tool_call["function"]["name"],
                    arguments=tool_call["function"]["arguments"],
                ),
            )
        )
        messages.append(tool_response)

    stream = call_llm(messages=messages, stream=True)
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
