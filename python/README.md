# python templates

## setup

For the Azure OpenAI examples, you need to have logged into the correct Azure subscription with [`az cli`](https://learn.microsoft.com/en-us/cli/azure/), and have a `Cognitive Services OpenAI User` or `Cognitive Services OpenAI Contributor` [role in Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/role-based-access-control).

```
# Create a virtual env and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Setup .env file
cp .env.example .env
# set the necessary values to .env

# Run an example, e.g.:
python openai/openai_basic.py

```

## Running the Ollama examples

In order to try the Ollama examples, first [get Ollama](https://ollama.com/)

```
# First download the model
ollama pull llama3.2

# Then run the example
python ollama/ollama_basic.py

```
