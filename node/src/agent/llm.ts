import fs from 'fs';
import path from 'path';
import { config } from 'dotenv';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';

config();

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

export class LLMInterface {
  private provider: string;
  private ignorePatterns: string[] = ['.env'];
  private client!: OpenAI | Anthropic;
  private model!: string;
  private tools!: any[];

  constructor(provider: string = 'openai') {
    this.provider = provider.toLowerCase();
    
    if (this.provider === 'openai') {
      this.setupOpenAI();
    } else if (this.provider === 'anthropic') {
      this.setupAnthropic();
    } else {
      throw new Error(`Unsupported provider: ${this.provider}`);
    }
  }

  private shouldIgnoreFile(filename: string): boolean {
    return this.ignorePatterns.some(pattern => filename.startsWith(pattern));
  }

  private setupOpenAI(): void {
    this.client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
    this.model = 'gpt-4o';
    this.tools = [
      {
        type: 'function',
        function: {
          name: 'list_files',
          description: 'List files and directories in a given path (excludes files starting with .env for security)',
          parameters: {
            type: 'object',
            properties: {
              path: {
                type: 'string',
                description: 'The directory path to list files from (default: current directory)',
              },
            },
            required: [],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'read_file',
          description: 'Read the contents of a file (cannot read files starting with .env for security)',
          parameters: {
            type: 'object',
            properties: {
              filepath: {
                type: 'string',
                description: 'The path to the file to read',
              },
            },
            required: ['filepath'],
          },
        },
      },
    ];
  }

  private setupAnthropic(): void {
    this.client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });
    this.model = 'claude-3-5-sonnet-20241022';
    this.tools = [
      {
        name: 'list_files',
        description: 'List files and directories in a given path (excludes files starting with .env for security)',
        input_schema: {
          type: 'object',
          properties: {
            path: {
              type: 'string',
              description: 'The directory path to list files from (default: current directory)',
            },
          },
          required: [],
        },
      },
      {
        name: 'read_file',
        description: 'Read the contents of a file (cannot read files starting with .env for security)',
        input_schema: {
          type: 'object',
          properties: {
            filepath: {
              type: 'string',
              description: 'The path to the file to read',
            },
          },
          required: ['filepath'],
        },
      },
    ];
  }

  async createCompletion(messages: Message[], systemPrompt: string): Promise<[string | null, ToolCall[] | null]> {
    if (this.provider === 'openai') {
      return this.createOpenAICompletion(messages, systemPrompt);
    } else {
      return this.createAnthropicCompletion(messages, systemPrompt);
    }
  }

  private async createOpenAICompletion(messages: Message[], systemPrompt: string): Promise<[string | null, ToolCall[] | null]> {
    const formattedMessages: any[] = [];
    
    if (systemPrompt) {
      formattedMessages.push({ role: 'system', content: systemPrompt });
    }

    for (const msg of messages) {
      if (msg.role === 'system') {
        continue; // Already added
      } else {
        formattedMessages.push(msg);
      }
    }

    const response = await (this.client as OpenAI).chat.completions.create({
      model: this.model,
      max_tokens: 1024,
      messages: formattedMessages,
      tools: this.tools,
    });

    const toolCalls = response.choices[0].message.tool_calls;
    if (toolCalls) {
      const unifiedToolCalls: ToolCall[] = toolCalls.map(tc => ({
        id: tc.id,
        name: tc.function.name,
        args: JSON.parse(tc.function.arguments),
      }));
      return [null, unifiedToolCalls];
    } else {
      return [response.choices[0].message.content, null];
    }
  }

  private async createAnthropicCompletion(messages: Message[], systemPrompt: string): Promise<[string | null, ToolCall[] | null]> {
    // Filter and convert messages to Anthropic format
    const userMessages = messages
      .filter(msg => msg.role !== 'system')
      .map(msg => ({
        role: msg.role as 'user' | 'assistant',
        content: msg.content,
      }));

    const response = await (this.client as Anthropic).messages.create({
      model: this.model,
      max_tokens: 1024,
      messages: userMessages,
      system: systemPrompt,
      tools: this.tools,
    });

    if (response.stop_reason === 'tool_use') {
      const textParts: string[] = [];
      const toolCalls: ToolCall[] = [];

      for (const block of response.content) {
        if (block.type === 'text') {
          textParts.push(block.text);
        } else if (block.type === 'tool_use') {
          toolCalls.push({
            id: block.id,
            name: block.name,
            args: block.input as Record<string, any>,
          });
        }
      }

      // Print any text that came with tool calls
      if (textParts.length > 0) {
        process.stdout.write(textParts.join(''));
      }

      return [null, toolCalls];
    } else {
      const textParts: string[] = [];
      for (const block of response.content) {
        if (block.type === 'text') {
          textParts.push(block.text);
        }
      }
      return [textParts.join(''), null];
    }
  }

  addToolResponse(messages: Message[], toolCall: ToolCall, result: string): void {
    if (this.provider === 'openai') {
      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        name: toolCall.name,
        content: result,
      });
    } else {
      // anthropic
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: toolCall.id,
            content: result,
          },
        ],
      });
    }
  }

  addAssistantMessageWithTools(messages: Message[], toolCalls: ToolCall[]): void {
    if (this.provider === 'openai') {
      const openaiToolCalls = toolCalls.map(tc => ({
        id: tc.id,
        type: 'function',
        function: {
          name: tc.name,
          arguments: JSON.stringify(tc.args),
        },
      }));

      messages.push({
        role: 'assistant',
        tool_calls: openaiToolCalls,
        content: '',
      });
    } else {
      // anthropic
      const content = toolCalls.map(tc => ({
        type: 'tool_use',
        id: tc.id,
        name: tc.name,
        input: tc.args,
      }));

      messages.push({
        role: 'assistant',
        content: content,
      });
    }
  }

  listFilesFiltered(dirPath: string = '.'): string {
    try {
      if (!fs.existsSync(dirPath)) {
        return `Error: Path '${dirPath}' does not exist`;
      }

      const items = fs.readdirSync(dirPath);
      if (items.length === 0) {
        return `Directory '${dirPath}' is empty`;
      }

      const files: string[] = [];
      const dirs: string[] = [];
      let filteredCount = 0;

      for (const item of items) {
        if (this.shouldIgnoreFile(item)) {
          filteredCount++;
          continue;
        }

        const itemPath = path.join(dirPath, item);
        if (fs.statSync(itemPath).isDirectory()) {
          dirs.push(`📁 ${item}/`);
        } else {
          files.push(`📄 ${item}`);
        }
      }

      let result = `Contents of '${dirPath}':\n`;
      if (dirs.length > 0) {
        result += '\nDirectories:\n' + dirs.join('\n');
      }
      if (files.length > 0) {
        result += '\nFiles:\n' + files.join('\n');
      }

      if (filteredCount > 0) {
        result += `\n\n(Hidden ${filteredCount} file(s) for security reasons)`;
      }

      return result;
    } catch (error) {
      return `Error listing files: ${error}`;
    }
  }

  readFileFiltered(filepath: string): string {
    try {
      const filename = path.basename(filepath);

      if (this.shouldIgnoreFile(filename)) {
        return `Error: Access to '${filepath}' is restricted for security reasons`;
      }

      if (!fs.existsSync(filepath)) {
        return `Error: File '${filepath}' does not exist`;
      }

      if (fs.statSync(filepath).isDirectory()) {
        return `Error: '${filepath}' is a directory, not a file`;
      }

      const content = fs.readFileSync(filepath, 'utf-8');
      return `Contents of '${filepath}':\n\`\`\`\n${content}\n\`\`\``;
    } catch (error) {
      return `Error reading file: ${error}`;
    }
  }
} 