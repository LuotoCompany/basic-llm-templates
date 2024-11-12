import ollama

client = ollama.Client()

stream = client.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Please write a poem"}],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
