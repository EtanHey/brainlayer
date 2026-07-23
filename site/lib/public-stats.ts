import {
  BRAINBAR_MCP_TOOL_COUNT,
  PYTHON_MCP_TOOL_COUNT,
} from "./public-tools";

// AIDEV-NOTE: Single checked-in snapshot for public site claims. Generate this
// file from the canonical DB + both MCP tool definitions to prevent stat rot.
// Snapshot verified 2026-07-23.
export const PUBLIC_SITE_STATS = {
  knowledgeChunks: 704_992,
  knowledgeChunksLabel: "700K+",
  knowledgeGraphEntities: 12_534,
  brainBarMcpTools: BRAINBAR_MCP_TOOL_COUNT,
  pythonMcpTools: PYTHON_MCP_TOOL_COUNT,
} as const;
