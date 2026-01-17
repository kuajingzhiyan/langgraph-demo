// src/models/openai.ts
import OpenAI from "openai";
import { BaseMessage, AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { getWriter, LangGraphRunnableConfig } from "@langchain/langgraph";

// 将 LangChain 消息转换为 OpenAI 格式
function convertToOpenAIMessages(messages: BaseMessage[]): OpenAI.ChatCompletionMessageParam[] {
  return messages.map((msg) => {
    if (msg instanceof HumanMessage) {
      return { role: "user" as const, content: msg.content as string };
    } else if (msg instanceof AIMessage) {
      // 如果 content 是数组，提取 text 部分
      const content = Array.isArray(msg.content)
        ? msg.content
          .filter((c): c is { type: "text"; text: string } => c.type === "text")
          .map((c) => c.text)
          .join("")
        : (msg.content as string);

      const result: OpenAI.ChatCompletionAssistantMessageParam = {
        role: "assistant" as const,
        content,
      };

      // 添加 tool_calls 支持
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        result.tool_calls = msg.tool_calls.map((tc) => ({
          id: tc.id!,
          type: "function" as const,
          function: {
            name: tc.name,
            arguments: JSON.stringify(tc.args),
          },
        }));
      }

      return result;
    } else if (msg instanceof ToolMessage) {
      // 处理工具响应消息
      return {
        role: "tool" as const,
        content: msg.content as string,
        tool_call_id: msg.tool_call_id!,
      };
    } else {
      return { role: "system" as const, content: msg.content as string };
    }
  });
}

// 使用 OpenAI 生成回复
export async function generateWithOpenAI(
  messages: BaseMessage[],
  config: LangGraphRunnableConfig,
  tools?: OpenAI.ChatCompletionTool[]
) {
  const client = new OpenAI({
    baseURL: process.env.OPENAI_BASE_URL,
    apiKey: process.env.OPENAI_API_KEY,
  });

  const modelName = process.env.MODEL_NAME || "gpt-4o";
  const writer = getWriter(config);
  const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const openaiMessages = convertToOpenAIMessages(messages);
  const response = await client.chat.completions.create({
    model: modelName,
    messages: openaiMessages,
    stream: true,
    tools: tools && tools.length > 0 ? tools : undefined,
  });

  let reasoningContent = "";
  let textContent = "";
  const toolCallsMap: Record<number, any> = {};

  for await (const chunk of response) {
    const delta = chunk.choices[0]?.delta as any;

    // 捕获 reasoning_content（思考内容）并立即发送事件
    if (delta?.reasoning_content) {
      reasoningContent += delta.reasoning_content;
      if (writer) {
        writer({
          type: "reasoning_chunk",
          message_id: messageId,
          content: delta.reasoning_content,
        });
      }
    }

    // 捕获普通 content（回复内容）并立即发送事件
    if (delta?.content) {
      textContent += delta.content;
      if (writer) {
        writer({
          type: "content_chunk",
          message_id: messageId,
          content: delta.content,
        });
      }
    }

    // 处理 tool_calls 流式数据
    if (delta?.tool_calls) {
      for (const toolCallDelta of delta.tool_calls) {
        const index = toolCallDelta.index;
        if (!toolCallsMap[index]) {
          toolCallsMap[index] = {
            id: toolCallDelta.id || "",
            type: "function" as const,
            function: {
              name: toolCallDelta.function?.name || "",
              arguments: toolCallDelta.function?.arguments || "",
            },
          };
        } else {
          if (toolCallDelta.function?.arguments) {
            toolCallsMap[index].function.arguments += toolCallDelta.function.arguments;
          }
        }
      }
    }
  }

  // 构建 tool_calls 数组（如果有）
  const toolCallsArray = Object.values(toolCallsMap);
  const formattedToolCalls = toolCallsArray.length > 0
    ? toolCallsArray.map((tc) => ({
        name: tc.function.name,
        args: JSON.parse(tc.function.arguments),
        id: tc.id,
      }))
    : undefined;

  return new AIMessage({
    id: messageId,
    content: textContent,
    tool_calls: formattedToolCalls,
    additional_kwargs: {
      reasoning_content: reasoningContent || undefined,
    },
  });
}
