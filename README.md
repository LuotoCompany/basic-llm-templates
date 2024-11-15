# basic-llm-templates

This repository contains simple templates for using LLM APIs in different languages and from different providers.

# content

## Node.js Examples

### OpenAI

- [Basic OpenAI Integration](node/src/openai/openai_basic.ts) - Simple example of using OpenAI API directly
- [OpenAI with Function Calling](node/src/openai/openai_tools.ts) - Example demonstrating OpenAI's function calling capabilities

### Azure OpenAI

- [Basic Azure OpenAI Integration](node/src/openai/azure_openai_basic.ts) - Simple example of using Azure OpenAI with Azure AD authentication
- [Azure OpenAI with Function Calling](node/src/openai/azure_openai_tools.ts) - Example demonstrating Azure OpenAI's function calling capabilities

### Anthropic

- [Basic Claude Integration](node/src/anthropic/claude_basic.ts) - Simple example of using Anthropic's Claude API
- [Claude with Tool Calling](node/src/anthropic/claude_tools.ts) - Example demonstrating Claude's tool calling capabilities

## Python Examples

### OpenAI

- [Basic OpenAI Integration](python/openai/openai_basic.py) - Simple example of using OpenAI API directly
- [OpenAI with Function Calling](python/openai/openai_tools.py) - Example demonstrating OpenAI's function calling capabilities
- [OpenAI with Function Calling and Streaming](python/openai/openai_tools_with_streaming.py) - Parsing multiple OpenAI tool calls while streaming the content

### Azure OpenAI

- [Basic Azure OpenAI Integration](python/openai/azure_openai_basic.py) - Simple example of using Azure OpenAI with Azure AD authentication
- [Azure OpenAI with Function Calling](python/openai/azure_openai_tools.py) - Example demonstrating Azure OpenAI's function calling capabilities
- [Azure OpenAI with Function Calling and Streaming](python/openai/azure_openai_tools_with_streaming.py) - Parsing multiple Azure OpenAI tool calls while streaming the content

### Anthropic

- [Basic Claude Integration](python/anthropic/claude_basic.py) - Simple example of using Anthropic's Claude API
- [Claude with Tool Calling](python/anthropic/claude_tools.py) - Example demonstrating Claude's tool calling capabilities

### Ollama

- [Basic Ollama Integration](python/ollama/ollama_basic.py) - Simple example of using Ollama's local LLM API
- [Ollama with Tool Calling](python/ollama/ollama_tools.py) - Example demonstrating Ollama's tool calling capabilities with a weather function
