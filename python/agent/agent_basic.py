import os
from llm import LLMInterface

# Configuration
LLM_PROVIDER = "openai"  # Can be "openai" or "anthropic"

SYS_PROMPT = """
You are a helpful agent that can read files and list directory contents. 
You have access to two tools:
1. list_files - to list files in a directory
2. read_file - to read the contents of a file

Use these tools when the user asks questions about files or directories.
"""


def list_files(path: str = ".") -> str:
    """List files and directories in the given path"""
    try:
        if not os.path.exists(path):
            return f"Error: Path '{path}' does not exist"

        items = os.listdir(path)
        if not items:
            return f"Directory '{path}' is empty"

        files = []
        dirs = []

        for item in items:
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

        return result
    except Exception as e:
        return f"Error listing files: {str(e)}"


def read_file_content(filepath: str) -> str:
    """Read the contents of a file"""
    try:
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' does not exist"

        if os.path.isdir(filepath):
            return f"Error: '{filepath}' is a directory, not a file"

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        return f"Contents of '{filepath}':\n```\n{content}\n```"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def execute_tool(tool_call: dict, llm: LLMInterface) -> str:
    """Execute a tool call and return the result"""
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print(f"🔧 Calling tool: {tool_name} with args: {tool_args}")

    if tool_name == "list_files":
        path = tool_args.get("path", ".")
        return llm.list_files_filtered(path)
    elif tool_name == "read_file":
        filepath = tool_args.get("filepath", "")
        return llm.read_file_filtered(filepath)
    else:
        return f"Error: Unknown tool '{tool_name}'"


def run_agent():
    """Main agent loop"""
    print(f"🤖 File Agent (using {LLM_PROVIDER.upper()}) - Ready to help!")
    print("Type 'quit' to exit")
    print("-" * 50)

    # Initialize LLM interface
    llm = LLMInterface(LLM_PROVIDER)

    # Initialize conversation
    messages = []
    waiting_for_user_input = True

    while True:
        if waiting_for_user_input:
            user_input = input("\n💬 You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            waiting_for_user_input = False

        print("\n🤖 Agent: ", end="", flush=True)

        # Get completion from LLM
        response_text, tool_calls = llm.create_completion(messages, SYS_PROMPT)

        if tool_calls:
            # Add assistant message with tool calls for OpenAI compatibility
            llm.add_assistant_message_with_tools(messages, tool_calls)

            # Execute each tool call
            for tool_call in tool_calls:
                result = execute_tool(tool_call, llm)
                llm.add_tool_response(messages, tool_call, result)

            # Continue the loop to get LLM response after tool calls
            continue
        else:
            # No tool calls, print the final response
            print(response_text)
            messages.append({"role": "assistant", "content": response_text})
            waiting_for_user_input = True


if __name__ == "__main__":
    run_agent()
