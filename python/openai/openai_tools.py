import json
import os
import random
from typing import List
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
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
    }
]


def handle_tool_call(tool_call: ChatCompletionMessageToolCall):
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    print(f"LLM called tool: {tool_name} with args: {tool_args}")

    temperature = random.randint(0, 20)
    weather_type = random.choice(["sunny", "cloudy", "raining"])

    if tool_name == "getWeather":
        city = tool_args["city"]
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": "getWeather",
            "content": f"The weather in {city} today: {temperature}°, {weather_type}",
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


user_msg = "What's the weather like in Helsinki?"

messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": user_msg},
]

response = call_llm(messages, stream=False, tools=tools)
tool_calls = response.choices[0].message.tool_calls

if tool_calls:
    for tool_call in tool_calls:
        messages.append({"role": "assistant", "tool_calls": [tool_call]})
        tool_response = handle_tool_call(tool_call=tool_call)
        messages.append(tool_response)

    stream = call_llm(messages=messages, stream=True)
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
else:
    print(response.choices[0].message.content)