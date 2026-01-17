// src/index.ts
import "dotenv/config";
import { StateGraph, MessagesAnnotation, START, END, LangGraphRunnableConfig, interrupt, isGraphInterrupt } from "@langchain/langgraph";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { generateWithAnthropic } from "./models/anthropic.js";
import { getMCPTools, executeMCPTool } from "./mcp/client.js";
import { createToolMessage, createErrorToolMessage } from "./mcp/tools.js";

type Decision =
  | { type: "approve" }
  | { type: "reject"; message?: string }
  | { type: "edit"; edited_action: { name: string; args: Record<string, unknown> } };

type HITLRequest = {
  action_requests: {
    name: string;
    args: Record<string, unknown>;
    description?: string;
  }[];
  review_configs: {
    action_name: string;
    allowed_decisions: Array<"approve" | "reject" | "edit">;
  }[];
};

const HITL_REJECT_MARKER = "[HITL_REJECTED]";

function isDeleteWindowTool(name: string): boolean {
  const normalized = name.toLowerCase();
  if (!normalized.startsWith("roxybrowser-openapi__")) return false;
  const isDeleteAction =
    normalized.includes("delete") || normalized.includes("remove");
  const isBrowserTarget =
    normalized.includes("browser") || normalized.includes("window");
  return isDeleteAction && isBrowserTarget;
}

function isRejectedToolMessage(message: unknown): boolean {
  if (!(message instanceof ToolMessage)) return false;
  if (typeof message.content !== "string") return false;
  return message.content.includes(HITL_REJECT_MARKER);
}

function buildDeleteInterrupt(toolCall: { name: string; args: Record<string, unknown> }): HITLRequest {
  return {
    action_requests: [
      {
        name: toolCall.name,
        args: toolCall.args,
        description: "This action will delete RoxyBrowser window(s). Please confirm before continuing.",
      },
    ],
    review_configs: [
      {
        action_name: toolCall.name,
        allowed_decisions: ["approve", "reject"],
      },
    ],
  };
}

function getDecision(resumeValue: unknown): Decision | undefined {
  if (!resumeValue || typeof resumeValue !== "object") return undefined;
  const decisions = (resumeValue as { decisions?: Decision[] }).decisions;
  if (!Array.isArray(decisions) || decisions.length === 0) return undefined;
  return decisions[0];
}

function createRejectionToolMessage(toolCallId: string, toolName: string): ToolMessage {
  return new ToolMessage({
    content: `${HITL_REJECT_MARKER} User rejected deletion request.`,
    tool_call_id: toolCallId,
    name: toolName,
    additional_kwargs: { error: true, hitl_rejected: true },
  });
}

// 对话节点
async function chat(
  state: typeof MessagesAnnotation.State,
  config: LangGraphRunnableConfig
) {
  const lastMessage = state.messages[state.messages.length - 1];
  if (isRejectedToolMessage(lastMessage)) {
    return {
      messages: [
        new AIMessage({
          content: "删除操作已被用户拒绝，我将停止执行此操作。如需继续，请重新发起请求。",
        }),
      ],
    };
  }

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

  const toolMessageIds = new Set(
    state.messages
      .filter((message) => (message as { tool_call_id?: string }).tool_call_id)
      .map((message) => (message as { tool_call_id?: string }).tool_call_id!)
  );

  const toolMessages: ToolMessage[] = [];
  for (const toolCall of toolCalls) {
    if (toolCall.id && toolMessageIds.has(toolCall.id)) {
      continue;
    }

    let args = toolCall.args as Record<string, unknown>;

    if (isDeleteWindowTool(toolCall.name)) {
      const resumeValue = interrupt(buildDeleteInterrupt({ name: toolCall.name, args }));
      const decision = getDecision(resumeValue);

      if (!decision || decision.type === "reject") {
        toolMessages.push(createRejectionToolMessage(toolCall.id!, toolCall.name));
        break;
      }

      if (decision.type === "edit" && decision.edited_action?.args) {
        args = decision.edited_action.args as Record<string, unknown>;
      }
    }

    try {
      const result = await executeMCPTool(toolCall.name, args);
      toolMessages.push(createToolMessage(result, toolCall.id!, toolCall.name));
    } catch (error) {
      if (isGraphInterrupt(error)) {
        throw error;
      }
      toolMessages.push(
        createErrorToolMessage(
          error instanceof Error ? error : new Error(String(error)),
          toolCall.id!,
          toolCall.name
        )
      );
    }
  }

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
