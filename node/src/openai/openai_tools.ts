import OpenAI from "openai";
import dotenv from "dotenv";
import path from "path";

import {
  ChatCompletionChunk,
  ChatCompletionTool,
} from "openai/resources/chat/completions";

import { Stream } from "openai/streaming";

dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const SYS_PROMPT = `
You are a friendly chatbot answering the user's questions.
`;

const tools: ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "getWeather",
      description: "Get weather information about a city",
      parameters: {
        type: "object",
        properties: {
          city: {
            type: "string",
            description: "Name of the city",
          },
        },
        required: ["city"],
      },
    },
  },
];

async function llmCall(
  messages: any[],
  stream = false,
  tools?: ChatCompletionTool[]
) {
  return await openai.chat.completions.create({
    model: "gpt-4o",
    messages: messages,
    temperature: 0.75,
    tools: tools,
    tool_choice: "auto",
    stream: stream,
  });
}

async function askAboutWeather() {
  const user_msg = "How's the weather in Helsinki?";

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: "system", content: SYS_PROMPT },
    { role: "user", content: user_msg },
  ];

  const response = (await llmCall(
    messages,
    false,
    tools
  )) as OpenAI.Chat.Completions.ChatCompletion;

  const tool_calls = response.choices[0].message.tool_calls;
  if (tool_calls) {
    for (const tool_call of tool_calls) {
      messages.push({
        role: "assistant",
        tool_calls: [tool_call],
      });

      const tool_name = tool_call.function.name;
      const tool_args = JSON.parse(tool_call.function.arguments);

      console.log(`LLM called tool: ${tool_name} with args:`, tool_args);

      if (tool_name === "getWeather") {
        const temperature = Math.floor(Math.random() * 21);
        const weather_types = ["sunny", "cloudy", "raining"];
        const weather_type =
          weather_types[Math.floor(Math.random() * weather_types.length)];

        const city = tool_args.city;
        messages.push({
          role: "tool",
          tool_call_id: tool_call.id,
          content: `The weather in ${city} today: ${temperature}°, ${weather_type}`,
        });
      } else {
        throw new Error(`Unknown tool ${tool_name}`);
      }
    }

    const stream = (await llmCall(
      messages,
      true,
      tools
    )) as Stream<OpenAI.Chat.Completions.ChatCompletionChunk>;

    for await (const part of stream) {
      process.stdout.write(part.choices[0]?.delta?.content || "");
    }
  }
}

askAboutWeather();
