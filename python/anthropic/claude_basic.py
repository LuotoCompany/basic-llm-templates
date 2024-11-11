import os
from anthropic import Anthropic

from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYS_PROMPT = """
You are a friendly chatbot answering the user's questions.
"""

user_msg = "Please write a short poem about LLMs"

with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": user_msg}],
    system=SYS_PROMPT,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
