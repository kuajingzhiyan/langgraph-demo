// src/index.ts
import "dotenv/config";
import { StateGraph, MessagesAnnotation, START, END, LangGraphRunnableConfig } from "@langchain/langgraph";
import { generateWithOpenAI } from "./models/openai.js";

// 对话节点
async function chat(
  state: typeof MessagesAnnotation.State,
  config: LangGraphRunnableConfig
) {
  const message = await generateWithOpenAI(state.messages, config);
  return { messages: [message] };
}

// 创建图
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("chat", chat)
  .addEdge(START, "chat")
  .addEdge("chat", END);

export const graph = workflow.compile();
