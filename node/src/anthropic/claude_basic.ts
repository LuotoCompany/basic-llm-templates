import Anthropic from "@anthropic-ai/sdk";
import { TextDelta } from "@anthropic-ai/sdk/resources/messages.mjs";
import dotenv from "dotenv";
import path from "path";

dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const anthropic = new Anthropic();

const SYS_PROMPT = `
You are a friendly chatbot answering the user's questions.
`;

async function writePoem() {
  const user_msg = "Please write a short poem about LLMs";

  const messages: { role: "user"; content: string }[] = [
    { role: "user", content: user_msg },
  ];

  const stream = anthropic.messages.stream({
    system: SYS_PROMPT,
    messages: messages,
    model: "claude-3-5-sonnet-20241022",
    max_tokens: 1024,
  });

  for await (const block of stream) {
    if (block.type === "content_block_delta") {
      process.stdout.write((block.delta as TextDelta).text || "");
    }
  }
}

writePoem();
