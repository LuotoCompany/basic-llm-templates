import * as readline from 'readline';
import { LLMInterface } from './llm';

// Configuration
const LLM_PROVIDER = 'anthropic'; // Can be "openai" or "anthropic"

const SYS_PROMPT = `
You are a helpful agent that can read files and list directory contents. 
You have access to two tools:
1. list_files - to list files in a directory
2. read_file - to read the contents of a file

Use these tools when the user asks questions about files or directories.
`;

interface Message {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string | any[];
  tool_calls?: any[];
  tool_call_id?: string;
  name?: string;
}

interface ToolCall {
  id: string;
  name: string;
  args: Record<string, any>;
}

function executeTool(toolCall: ToolCall, llm: LLMInterface): string {
  const toolName = toolCall.name;
  const toolArgs = toolCall.args;

  console.log(`🔧 Calling tool: ${toolName} with args:`, toolArgs);

  if (toolName === 'list_files') {
    const path = toolArgs.path || '.';
    return llm.listFilesFiltered(path);
  } else if (toolName === 'read_file') {
    const filepath = toolArgs.filepath || '';
    return llm.readFileFiltered(filepath);
  } else {
    return `Error: Unknown tool '${toolName}'`;
  }
}

async function runAgent(): Promise<void> {
  console.log(`🤖 File Agent (using ${LLM_PROVIDER.toUpperCase()}) - Ready to help!`);
  console.log("Type 'quit' to exit");
  console.log('-'.repeat(50));

  // Initialize LLM interface
  const llm = new LLMInterface(LLM_PROVIDER);

  // Initialize conversation
  const messages: Message[] = [];
  let waitingForUserInput = true;

  // Setup readline interface
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const askQuestion = (query: string): Promise<string> => {
    return new Promise((resolve) => {
      rl.question(query, (answer) => {
        resolve(answer);
      });
    });
  };

  try {
    while (true) {
      if (waitingForUserInput) {
        const userInput = await askQuestion('\n💬 You: ');

        if (['quit', 'exit', 'q'].includes(userInput.toLowerCase().trim())) {
          console.log('👋 Goodbye!');
          break;
        }

        if (!userInput.trim()) {
          continue;
        }

        messages.push({ role: 'user', content: userInput.trim() });
        waitingForUserInput = false;
      }

      process.stdout.write('\n🤖 Agent: ');

      // Get completion from LLM
      const [responseText, toolCalls] = await llm.createCompletion(messages, SYS_PROMPT);

      if (toolCalls) {
        // Add assistant message with tool calls for OpenAI compatibility
        llm.addAssistantMessageWithTools(messages, toolCalls);

        // Execute each tool call
        for (const toolCall of toolCalls) {
          const result = executeTool(toolCall, llm);
          llm.addToolResponse(messages, toolCall, result);
        }

        // Continue the loop to get LLM response after tool calls
        continue;
      } else {
        // No tool calls, print the final response
        console.log(responseText);
        messages.push({ role: 'assistant', content: responseText || '' });
        waitingForUserInput = true;
      }
    }
  } catch (error) {
    console.error('Error:', error);
  } finally {
    rl.close();
  }
}

// Only run if this file is executed directly
if (require.main === module) {
  runAgent().catch(console.error);
}

export { runAgent }; 