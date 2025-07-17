import json
import os
from typing import Any, List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


class LLMInterface:
    """Unified interface for different LLM providers"""

    def __init__(self, provider: str = "openai"):
        self.provider = provider.lower()
        # Files/patterns to ignore for security
        self.ignore_patterns = [".env"]
        if self.provider == "openai":
            self._setup_openai()
        elif self.provider == "anthropic":
            self._setup_anthropic()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _should_ignore_file(self, filename: str) -> bool:
        """Check if a file should be ignored based on ignore patterns"""
        for pattern in self.ignore_patterns:
            if filename.startswith(pattern):
                return True
        return False

    def _setup_openai(self):
        """Setup OpenAI client and tools"""
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files and directories in a given path (excludes files starting with .env for security)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The directory path to list files from (default: current directory)",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file (cannot read files starting with .env for security)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "The path to the file to read",
                            }
                        },
                        "required": ["filepath"],
                    },
                },
            },
        ]

    def _setup_anthropic(self):
        """Setup Anthropic client and tools"""
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-3-5-sonnet-20241022"
        self.tools = [
            {
                "name": "list_files",
                "description": "List files and directories in a given path (excludes files starting with .env for security)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory path to list files from (default: current directory)",
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "read_file",
                "description": "Read the contents of a file (cannot read files starting with .env for security)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    },
                    "required": ["filepath"],
                },
            },
        ]

    def create_completion(
        self, messages: List[Dict], system_prompt: str
    ) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """
        Create a completion and return (response_text, tool_calls)
        Returns (None, tool_calls) if tools need to be executed
        Returns (response_text, None) if no tools needed
        """
        if self.provider == "openai":
            return self._create_openai_completion(messages, system_prompt)
        else:
            return self._create_anthropic_completion(messages, system_prompt)

    def _create_openai_completion(
        self, messages: List[Dict], system_prompt: str
    ) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """Handle OpenAI completion"""
        # Convert messages to proper format for OpenAI
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg["role"] == "system":
                continue  # Already added
            else:
                formatted_messages.append(msg)

        # Type ignore since we know self.client is OpenAI client in this context
        response = self.client.chat.completions.create(  # type: ignore
            model=self.model,
            max_tokens=1024,
            messages=formatted_messages,  # type: ignore
            tools=self.tools,  # type: ignore
        )

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            # Convert OpenAI tool calls to unified format
            unified_tool_calls = []
            for tc in tool_calls:
                unified_tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments),
                    }
                )
            return None, unified_tool_calls
        else:
            return response.choices[0].message.content, None

    def _create_anthropic_completion(
        self, messages: List[Dict], system_prompt: str
    ) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """Handle Anthropic completion"""
        # Filter out system messages from the message list
        user_messages = [msg for msg in messages if msg["role"] != "system"]

        # Type ignore since we know self.client is Anthropic client in this context
        response = self.client.messages.create(  # type: ignore
            model=self.model,
            max_tokens=1024,
            messages=user_messages,  # type: ignore
            system=system_prompt,
            tools=self.tools,  # type: ignore
        )

        if response.stop_reason == "tool_use":
            # Extract tool calls and any text
            text_parts = []
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(
                        {"id": block.id, "name": block.name, "args": block.input}
                    )

            # Print any text that came with tool calls
            if text_parts:
                print("".join(text_parts), end="", flush=True)

            return None, tool_calls
        else:
            # Extract text from response
            text_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)

            return "".join(text_parts), None

    def add_tool_response(
        self, messages: List[Dict], tool_call: Dict, result: str
    ) -> None:
        """Add a tool response to the message history in the correct format"""
        if self.provider == "openai":
            # Add the tool result
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["name"],
                    "content": result,
                }
            )
        else:  # anthropic
            # For Anthropic, we need to add the tool result as a user message
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call["id"],
                            "content": result,
                        }
                    ],
                }
            )

    def add_assistant_message_with_tools(
        self, messages: List[Dict], tool_calls: List[Dict]
    ) -> None:
        """Add assistant message with tool calls (needed for OpenAI format)"""
        if self.provider == "openai":
            # Convert unified tool calls back to OpenAI format
            openai_tool_calls = []
            for tc in tool_calls:
                openai_tool_calls.append(
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                )

            messages.append({"role": "assistant", "tool_calls": openai_tool_calls})
        else:  # anthropic
            # For Anthropic, we need to add the assistant message with tool_use blocks
            content = []
            for tc in tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    }
                )

            messages.append({"role": "assistant", "content": content})

    def list_files_filtered(self, path: str = ".") -> str:
        """List files and directories with filtering applied"""
        try:
            if not os.path.exists(path):
                return f"Error: Path '{path}' does not exist"

            items = os.listdir(path)
            if not items:
                return f"Directory '{path}' is empty"

            files = []
            dirs = []
            filtered_count = 0

            for item in items:
                # Filter out ignored files
                if self._should_ignore_file(item):
                    filtered_count += 1
                    continue

                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    dirs.append(f"📁 {item}/")
                else:
                    files.append(f"📄 {item}")

            result = f"Contents of '{path}':\n"
            if dirs:
                result += "\nDirectories:\n" + "\n".join(dirs)
            if files:
                result += "\nFiles:\n" + "\n".join(files)

            if filtered_count > 0:
                result += f"\n\n(Hidden {filtered_count} file(s) for security reasons)"

            return result
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def read_file_filtered(self, filepath: str) -> str:
        """Read file contents with filtering applied"""
        try:
            # Extract filename from path for checking
            filename = os.path.basename(filepath)

            # Check if file should be ignored
            if self._should_ignore_file(filename):
                return (
                    f"Error: Access to '{filepath}' is restricted for security reasons"
                )

            if not os.path.exists(filepath):
                return f"Error: File '{filepath}' does not exist"

            if os.path.isdir(filepath):
                return f"Error: '{filepath}' is a directory, not a file"

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            return f"Contents of '{filepath}':\n```\n{content}\n```"
        except Exception as e:
            return f"Error reading file: {str(e)}"
