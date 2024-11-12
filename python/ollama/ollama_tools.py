import ollama
import random
import asyncio

MODEL = "llama3.2"

SYS_PROMPT = """
You are a friendly chatbot answering the user's questions.
"""

tools = [
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


def handle_tool_call(tool_call):
    tool_name = tool_call["function"]["name"]
    tool_args = tool_call["function"]["arguments"]

    if tool_name == "getWeather":
        temperature = random.randint(0, 20)
        weather_type = random.choice(["sunny", "cloudy", "raining"])
        city = tool_args["city"]
        return f"The weather in {city} today: {temperature}°, {weather_type}"
    else:
        raise ValueError(f"Unknown tool {tool_name}")


async def run():
    client = ollama.AsyncClient()

    user_msg = "What's the weather like in Helsinki?"

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    response = await client.chat(model=MODEL, messages=messages, tools=tools)
    messages.append(response["message"])

    if response["message"].get("tool_calls"):
        for tool_call in response["message"]["tool_calls"]:
            tool_result = handle_tool_call(tool_call)
            messages.append({"role": "tool", "content": tool_result})

    stream = await client.chat(
        model=MODEL, messages=messages, tools=tools, stream=True
    )

    async for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)


asyncio.run(run())
