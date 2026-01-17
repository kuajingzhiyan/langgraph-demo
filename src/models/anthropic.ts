import { AIMessage, BaseMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { getWriter, LangGraphRunnableConfig } from "@langchain/langgraph";

type AnthropicMessage = {
  role: "user" | "assistant";
  content: string | Array<Record<string, any>>;
};

type OpenAITool = {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, any>;
  };
};

type AnthropicTool = {
  name: string;
  description?: string;
  input_schema?: Record<string, any>;
};

type ToolCallState = {
  id: string;
  name: string;
  input: string;
};

function getMessageType(message: BaseMessage): string | undefined {
  const maybe = message as unknown as {
    _getType?: () => string;
    getType?: () => string;
  };
  if (typeof maybe._getType === "function") return maybe._getType();
  if (typeof maybe.getType === "function") return maybe.getType();
  return undefined;
}

function getTextContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((c): c is { type: "text"; text: string } => c?.type === "text")
    .map((c) => c.text)
    .join("");
}

function toAnthropicTools(tools?: OpenAITool[]): AnthropicTool[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  return tools
    .filter((tool) => tool?.type === "function" && tool.function?.name)
    .map((tool) => ({
      name: tool.function.name,
      description: tool.function.description || "",
      input_schema: tool.function.parameters || { type: "object", properties: {} },
    }));
}

function buildAssistantContent(
  message: AIMessage,
  thinkingEnabled: boolean
): Array<Record<string, any>> {
  const blocks: Array<Record<string, any>> = [];
  const reasoningContent = (message as Record<string, unknown>).additional_kwargs as
    | Record<string, unknown>
    | undefined;
  const reasoningText = (reasoningContent?.reasoning_content as string | undefined) ?? "";

  if (thinkingEnabled) {
    blocks.push({ type: "thinking", thinking: reasoningText });
  }

  const text = getTextContent(message.content);
  if (text) {
    blocks.push({ type: "text", text });
  }

  if (message.tool_calls && message.tool_calls.length > 0) {
    for (const tc of message.tool_calls) {
      blocks.push({
        type: "tool_use",
        id: tc.id,
        name: tc.name,
        input: tc.args ?? {},
      });
    }
  }

  if (blocks.length === 0) {
    blocks.push({ type: "text", text: "" });
  }

  return blocks;
}

function buildAnthropicMessages(
  messages: BaseMessage[],
  thinkingEnabled: boolean
): { messages: AnthropicMessage[]; system?: string } {
  const systemParts: string[] = [];
  const out: AnthropicMessage[] = [];

  for (const msg of messages) {
    const msgType = getMessageType(msg);
    if (msgType === "system") {
      const text = getTextContent(msg.content);
      if (text) systemParts.push(text);
      continue;
    }

    if (msg instanceof HumanMessage) {
      out.push({ role: "user", content: getTextContent(msg.content) });
    } else if (msg instanceof AIMessage) {
      out.push({
        role: "assistant",
        content: buildAssistantContent(msg, thinkingEnabled),
      });
    } else if (msg instanceof ToolMessage) {
      out.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: msg.tool_call_id!,
            content: msg.content as string,
          },
        ],
      });
    }
  }

  return {
    messages: out,
    system: systemParts.length > 0 ? systemParts.join("\n") : undefined,
  };
}

function parseEventPayload(raw: string): { event: string; data: string } | null {
  const lines = raw.split("\n");
  let event = "message";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }
  const data = dataLines.join("\n");
  if (!data) return null;
  return { event, data };
}

function safeJsonParse<T>(value: string): T | undefined {
  try {
    return JSON.parse(value) as T;
  } catch {
    return undefined;
  }
}

function parseToolArgs(input: string): Record<string, any> {
  if (!input) return {};
  const parsed = safeJsonParse<Record<string, any>>(input);
  if (parsed) return parsed;
  return {};
}

function getToolInputString(input: unknown): string {
  if (!input) return "";
  if (typeof input === "string") return input;
  if (typeof input !== "object" || Array.isArray(input)) return "";
  const keys = Object.keys(input as Record<string, unknown>);
  if (keys.length === 0) return "";
  return JSON.stringify(input);
}

// 使用 Anthropic /v1/messages 生成回复
export async function generateWithAnthropic(
  messages: BaseMessage[],
  config: LangGraphRunnableConfig,
  tools?: OpenAITool[]
) {
  const apiKey = process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY || "";
  const rawBaseUrl = process.env.ANTHROPIC_BASE_URL || process.env.OPENAI_BASE_URL || "";
  const modelName = process.env.MODEL_NAME || "claude-sonnet-4-5";
  const thinkingEnabled =
    process.env.THINKING_ENABLED === "true" ||
    (process.env.MODEL_NAME || "").includes("thinking");
  const thinkingBudget = Number(process.env.THINKING_BUDGET_TOKENS || "10000");
  const maxTokens = Number(process.env.MAX_TOKENS || "2048");

  const writer = getWriter(config);
  const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const { messages: anthropicMessages, system } = buildAnthropicMessages(
    messages,
    thinkingEnabled
  );

  const payload: Record<string, any> = {
    model: modelName.replace(/-thinking$/, ""),
    max_tokens: maxTokens,
    stream: true,
    messages: anthropicMessages,
    tools: toAnthropicTools(tools),
  };

  if (system) payload.system = system;
  if (thinkingEnabled) {
    payload.thinking = { type: "enabled", budget_tokens: thinkingBudget };
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "anthropic-version": process.env.ANTHROPIC_VERSION || "2023-06-01",
  };
  if (apiKey) {
    headers["x-api-key"] = apiKey;
    headers["authorization"] = `Bearer ${apiKey}`;
  }

  const normalizedBaseUrl = rawBaseUrl.replace(/\/$/, "");
  const endpoint = normalizedBaseUrl.endsWith("/v1")
    ? `${normalizedBaseUrl}/messages`
    : `${normalizedBaseUrl}/v1/messages`;

  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Anthropic request failed: ${response.status} ${errorText}`);
  }

  if (!response.body) {
    throw new Error("Anthropic response body is empty");
  }

  let reasoningContent = "";
  let textContent = "";
  const toolCallsMap: Record<number, ToolCallState> = {};

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    while (true) {
      const boundaryIndex = buffer.indexOf("\n\n");
      if (boundaryIndex === -1) break;
      const chunk = buffer.slice(0, boundaryIndex);
      buffer = buffer.slice(boundaryIndex + 2);

      const parsed = parseEventPayload(chunk);
      if (!parsed || parsed.data === "[DONE]") continue;
      const payload = safeJsonParse<Record<string, any>>(parsed.data);
      if (!payload) continue;

      if (parsed.event === "error") {
        const errorMessage =
          payload.error?.message || payload.message || "Unknown streaming error";
        throw new Error(errorMessage);
      }

      if (parsed.event === "content_block_start") {
        const block = payload.content_block || {};
        const index = payload.index as number;
        if (block.type === "text" && block.text) {
          textContent += block.text;
          if (writer) {
            writer({
              type: "content_chunk",
              message_id: messageId,
              content: block.text,
            });
          }
        } else if (
          (block.type === "thinking" || block.type === "redacted_thinking") &&
          (block.thinking || block.text)
        ) {
          const thinkingText = block.thinking || block.text || "";
          reasoningContent += thinkingText;
          if (writer) {
            writer({
              type: "reasoning_chunk",
              message_id: messageId,
              content: thinkingText,
            });
          }
        } else if (block.type === "tool_use") {
          toolCallsMap[index] = {
            id: block.id || "",
            name: block.name || "",
            input: getToolInputString(block.input),
          };
        }
      }

      if (parsed.event === "content_block_delta") {
        const index = payload.index as number;
        const delta = payload.delta || {};
        if (delta.type === "text_delta" && delta.text) {
          textContent += delta.text;
          if (writer) {
            writer({
              type: "content_chunk",
              message_id: messageId,
              content: delta.text,
            });
          }
        } else if (
          (delta.type === "thinking_delta" || delta.type === "redacted_thinking_delta") &&
          (delta.thinking || delta.text)
        ) {
          const thinkingText = delta.thinking || delta.text || "";
          reasoningContent += thinkingText;
          if (writer) {
            writer({
              type: "reasoning_chunk",
              message_id: messageId,
              content: thinkingText,
            });
          }
        } else if (delta.type === "input_json_delta" && delta.partial_json) {
          if (!toolCallsMap[index]) {
            toolCallsMap[index] = { id: "", name: "", input: "" };
          }
          toolCallsMap[index].input += delta.partial_json;
        }
      }
    }
  }

  const formattedToolCalls = Object.entries(toolCallsMap)
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(([_, tc]) => ({
      name: tc.name,
      args: parseToolArgs(tc.input),
      id: tc.id,
    }));

  return new AIMessage({
    id: messageId,
    content: textContent,
    tool_calls: formattedToolCalls.length > 0 ? formattedToolCalls : undefined,
    additional_kwargs: {
      reasoning_content: reasoningContent || undefined,
    },
  });
}
