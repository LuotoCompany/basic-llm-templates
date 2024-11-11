import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYS_PROMPT = """
You are a friendly chatbot answering the user's questions.
"""

user_msg = "Please write a short poem about LLMs"

messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": user_msg},
]


stream = llm.chat.completions.create(
    model="gpt-4o", max_tokens=1024, messages=messages, stream=True
)

for chunk in stream:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
