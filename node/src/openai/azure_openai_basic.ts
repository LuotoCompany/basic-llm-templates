import OpenAI, { AzureOpenAI } from "openai";
import dotenv from "dotenv";
import path from "path";
import {
  DefaultAzureCredential,
  getBearerTokenProvider,
} from "@azure/identity";

const credential = new DefaultAzureCredential();
const scope = "https://cognitiveservices.azure.com/.default";
const azureADTokenProvider = getBearerTokenProvider(credential, scope);

dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const openai = new AzureOpenAI({
  apiVersion: process.env.AZURE_OPENAI_API_VERSION,
  baseURL: `${process.env.AZURE_OPENAI_API_BASE_URL}/openai/deployments/${process.env.AZURE_OPENAI_DEPLOYMENT_NAME}`,
  azureADTokenProvider,
});

const SYS_PROMPT = `
You are a friendly chatbot answering the user's questions.
`;

async function writePoem() {
  const user_msg = "Please write a short poem about LLMs";

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: "system", content: SYS_PROMPT },
    { role: "user", content: user_msg },
  ];

  const stream = await openai.chat.completions.create({
    messages: messages,
    model: "gpt-4o",
    stream: true,
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
  }
}

writePoem();
