import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

credential = DefaultAzureCredential()
AZURE_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
azure_token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), AZURE_COGNITIVE_SERVICES_SCOPE
)

llm = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE_URL"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_ad_token_provider=azure_token_provider,
)

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
    if len(chunk.choices) > 0 and chunk.choices[0].delta:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
