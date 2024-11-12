import Anthropic from "@anthropic-ai/sdk";
import { Tool } from "@anthropic-ai/sdk/resources/messages.mjs";
import dotenv from "dotenv";
import path from "path";

dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const anthropic = new Anthropic();

const MODEL = "claude-3-5-sonnet-20241022";

const SYS_PROMPT = `
You are a friendly chatbot answering the user's questions.
`;

const tools: Tool[] = [
  {
    name: "getWeather",
    description: "Get weather information about a city",
    input_schema: {
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
];

const user_msg = "What's the weather like in Helsinki?";

const messages: Anthropic.Messages.MessageParam[] = [
  { role: "user", content: user_msg },
];

const handleToolCall = (toolCall: Anthropic.Messages.ToolUseBlock) => {
  const toolArgs = toolCall.input as { city: string };

  if (toolCall.name === "getWeather") {
    const temperature = Math.floor(Math.random() * 21);
    const weather_types = ["sunny", "cloudy", "raining"];
    const weather_type =
      weather_types[Math.floor(Math.random() * weather_types.length)];

    const city = toolArgs.city;

    const toolResult: Anthropic.Messages.ToolResultBlockParam = {
      type: "tool_result",
      tool_use_id: toolCall.id,
      content: `The weather in ${city} today: ${temperature}°, ${weather_type}`,
    };
    return toolResult;
  } else {
    throw new Error(`Unknown tool ${toolCall.name}`);
  }
};

const askWeather = async () => {
  const response = await anthropic.messages.create({
    system: SYS_PROMPT,
    messages: messages,
    model: MODEL,
    max_tokens: 1024,
    tools: tools,
  });

  if (response.stop_reason == "end_turn") {
    if (response.content[0].type == "text") {
      console.log(response.content[0].text);
    }
  } else if (response.stop_reason == "tool_use") {
    messages.push({ role: "assistant", content: response.content });

    const blocks: Anthropic.Messages.ContentBlock[] = response.content;

    for (const block of blocks) {
      if (typeof block === "string") {
        console.log(block);
      } else if (block.type === "text") {
        console.log(block.text);
      } else if (block.type === "tool_use") {
        // Handle tool use block
        console.log(`Tool called: ${block.name}`);
        console.log(`Tool input: ${JSON.stringify(block.input)}`);

        const tool_result = handleToolCall(block);
        messages.push({ role: "user", content: [tool_result] });
      }
    }

    const stream = anthropic.messages.stream({
      system: SYS_PROMPT,
      messages,
      model: MODEL,
      max_tokens: 1024,
      tools: tools,
    });

    for await (const block of stream) {
      if (block.type === "content_block_delta") {
        process.stdout.write(
          (block.delta as Anthropic.Messages.TextDelta).text || ""
        );
      }
    }
  }
};

askWeather();
