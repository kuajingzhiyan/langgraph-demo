import { ToolMessage } from "@langchain/core/messages";

/**
 * Format MCP tool result as a string for ToolMessage content
 */
export function formatToolResult(result: any): string {
  if (!result) {
    return "Tool executed successfully with no output";
  }

  // If result has content array (MCP standard format)
  if (result.content && Array.isArray(result.content)) {
    return result.content
      .map((item: any) => {
        if (item.type === "text") {
          return item.text;
        } else if (item.type === "image") {
          return `[Image: ${item.mimeType || "unknown"}]`;
        } else if (item.type === "resource") {
          return `[Resource: ${item.uri || "unknown"}]`;
        }
        return JSON.stringify(item);
      })
      .join("\n");
  }

  // If result is a simple value
  if (typeof result === "string") {
    return result;
  }

  // Otherwise, stringify it
  return JSON.stringify(result, null, 2);
}

/**
 * Create a ToolMessage from an MCP tool execution result
 */
export function createToolMessage(
  result: any,
  toolCallId: string,
  toolName: string
): ToolMessage {
  const content = formatToolResult(result);

  return new ToolMessage({
    content,
    tool_call_id: toolCallId,
    name: toolName,
  });
}

/**
 * Create an error ToolMessage from a failed tool execution
 */
export function createErrorToolMessage(
  error: Error,
  toolCallId: string,
  toolName: string
): ToolMessage {
  return new ToolMessage({
    content: `Error executing tool: ${error.message}`,
    tool_call_id: toolCallId,
    name: toolName,
    additional_kwargs: { error: true },
  });
}
