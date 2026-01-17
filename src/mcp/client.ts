import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

interface MCPServerConfig {
  command: string;
  args: string[];
  env?: Record<string, string>;
}

interface MCPConfig {
  mcpServers: Record<string, MCPServerConfig>;
}

interface MCPClientInstance {
  client: Client;
  serverName: string;
}

// Singleton state
let mcpClients: MCPClientInstance[] = [];
let isInitialized = false;

/**
 * Initialize all MCP clients from configuration file
 */
export async function initializeMCPClient(): Promise<void> {
  if (isInitialized) {
    return;
  }

  try {
    // Read configuration file
    const configPath = join(__dirname, "../../mcp-service.json");
    let config: MCPConfig;

    try {
      const configContent = readFileSync(configPath, "utf-8");
      config = JSON.parse(configContent);
    } catch (error) {
      console.log("No mcp-service.json found or invalid JSON. Running without MCP tools.");
      isInitialized = true;
      return;
    }

    if (!config.mcpServers || Object.keys(config.mcpServers).length === 0) {
      console.log("No MCP servers configured. Running without MCP tools.");
      isInitialized = true;
      return;
    }

    // Initialize each server
    const initPromises = Object.entries(config.mcpServers).map(
      async ([serverName, serverConfig]) => {
        try {
          console.log(`Initializing MCP server: ${serverName}`);

          const transport = new StdioClientTransport({
            command: serverConfig.command,
            args: serverConfig.args,
            env: {
              ...(Object.fromEntries(
                Object.entries(process.env).filter(([_, v]) => v !== undefined)
              ) as Record<string, string>),
              ...serverConfig.env,
            },
          });

          const client = new Client(
            {
              name: "langgraph-agent",
              version: "1.0.0",
            },
            {
              capabilities: {},
            }
          );

          await client.connect(transport);
          console.log(`✓ Connected to MCP server: ${serverName}`);

          mcpClients.push({ client, serverName });
        } catch (error) {
          console.warn(
            `Failed to initialize MCP server ${serverName}:`,
            error instanceof Error ? error.message : String(error)
          );
        }
      }
    );

    await Promise.all(initPromises);
    isInitialized = true;

    console.log(`Initialized ${mcpClients.length} MCP server(s)`);
  } catch (error) {
    console.error("Error initializing MCP clients:", error);
    isInitialized = true; // Mark as initialized to prevent retries
  }
}

/**
 * Get all available MCP tools in OpenAI format
 */
export async function getMCPTools(): Promise<any[]> {
  // Initialize if not already done
  if (!isInitialized) {
    await initializeMCPClient();
  }

  if (mcpClients.length === 0) {
    return [];
  }

  const allTools: any[] = [];

  for (const { client, serverName } of mcpClients) {
    try {
      const { tools } = await client.listTools();

      for (const tool of tools) {
        // Convert MCP tool to OpenAI format
        allTools.push({
          type: "function" as const,
          function: {
            name: `${serverName}__${tool.name}`,
            description: tool.description || "",
            parameters: tool.inputSchema || { type: "object", properties: {} },
          },
        });
      }
    } catch (error) {
      console.warn(
        `Failed to list tools from ${serverName}:`,
        error instanceof Error ? error.message : String(error)
      );
    }
  }

  return allTools;
}

/**
 * Execute an MCP tool call
 */
export async function executeMCPTool(
  toolName: string,
  args: Record<string, any>
): Promise<any> {
  // Parse server name and tool name
  const parts = toolName.split("__");
  if (parts.length !== 2) {
    throw new Error(`Invalid tool name format: ${toolName}`);
  }

  const [serverName, actualToolName] = parts;

  // Find the client for this server
  const clientInstance = mcpClients.find((c) => c.serverName === serverName);
  if (!clientInstance) {
    throw new Error(`MCP server not found: ${serverName}`);
  }

  try {
    const result = await clientInstance.client.callTool({
      name: actualToolName,
      arguments: args,
    });

    return result;
  } catch (error) {
    throw new Error(
      `Tool execution failed: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Close all MCP client connections
 */
export async function closeMCPClient(): Promise<void> {
  for (const { client, serverName } of mcpClients) {
    try {
      await client.close();
      console.log(`Closed MCP server: ${serverName}`);
    } catch (error) {
      console.warn(
        `Failed to close MCP server ${serverName}:`,
        error instanceof Error ? error.message : String(error)
      );
    }
  }

  mcpClients = [];
  isInitialized = false;
}
