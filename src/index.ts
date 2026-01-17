// src/index.ts
import "dotenv/config";
import { StateGraph, MessagesAnnotation, START, END, LangGraphRunnableConfig } from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";
import { generateWithAnthropic } from "./models/anthropic.js";
import { getMCPTools, executeMCPTool } from "./mcp/client.js";
import { createToolMessage, createErrorToolMessage } from "./mcp/tools.js";

// 对话节点
async function chat(
  state: typeof MessagesAnnotation.State,
  config: LangGraphRunnableConfig
) {
  // 获取 MCP 工具（首次调用时初始化，之后使用缓存）
  const mcpTools = await getMCPTools();

  const message = await generateWithAnthropic(state.messages, config, mcpTools);
  return { messages: [message] };
}

// 工具执行节点
async function tools(
  state: typeof MessagesAnnotation.State,
  config: LangGraphRunnableConfig
) {
  const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
  const toolCalls = lastMessage.tool_calls || [];

  // 并行执行所有工具调用
  const toolMessages = await Promise.all(
    toolCalls.map(async (toolCall) => {
      try {
        const result = await executeMCPTool(toolCall.name, toolCall.args);
        return createToolMessage(result, toolCall.id!, toolCall.name);
      } catch (error) {
        return createErrorToolMessage(
          error instanceof Error ? error : new Error(String(error)),
          toolCall.id!,
          toolCall.name
        );
      }
    })
  );

  return { messages: toolMessages };
}

// 条件路由函数
function shouldContinue(
  state: typeof MessagesAnnotation.State
): "tools" | typeof END {
  const lastMessage = state.messages[state.messages.length - 1];

  // 类型守卫
  if (!("tool_calls" in lastMessage)) {
    return END;
  }

  const aiMessage = lastMessage as AIMessage;
  return aiMessage.tool_calls && aiMessage.tool_calls.length > 0 ? "tools" : END;
}

// 创建图
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("chat", chat)
  .addNode("tools", tools)
  .addEdge(START, "chat")
  .addConditionalEdges("chat", shouldContinue, {
    tools: "tools",
    [END]: END,
  })
  .addEdge("tools", "chat");

export const graph = workflow.compile();
