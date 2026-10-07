import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import {
  BRAINBAR_MCP_TOOL_COUNT,
  BRAINBAR_MCP_TOOL_GROUPS,
  PYTHON_MCP_TOOL_COUNT,
} from "../site/lib/public-tools";
import { PUBLIC_SITE_STATS } from "../site/lib/public-stats";

const routerSource = readFileSync(
  new URL("../brain-bar/Sources/BrainBar/MCPRouter.swift", import.meta.url),
  "utf8",
).split("static let toolDefinitions")[1];
const routerNames = [...routerSource.matchAll(/"name": "(brain_[a-z_]+)"/g)]
  .map((match) => match[1]).sort();
const siteNames: string[] = BRAINBAR_MCP_TOOL_GROUPS.flatMap((group) =>
  group.tools.map((tool) => tool.name),
).sort();

test("public site inventory matches all 16 canonical native tools", () => {
  expect(routerNames).toHaveLength(16);
  expect(BRAINBAR_MCP_TOOL_COUNT).toBe(routerNames.length);
  expect(PUBLIC_SITE_STATS.brainBarMcpTools).toBe(routerNames.length);
  expect(siteNames).toEqual(routerNames);
});

test("public site does not advertise retired tools", () => {
  expect(siteNames).not.toContain("brain_enrich");
});

const pythonSource = readFileSync(
  new URL("../src/brainlayer/mcp/__init__.py", import.meta.url), "utf8",
).split("def _full_tool_definitions()")[1].split("async def list_tools()")[0];
const pythonNames = [...pythonSource.matchAll(/name="(brain_[a-z_]+)"/g)]
  .map((match) => match[1]).sort();

test("public Python count matches the library canonical inventory", () => {
  expect(pythonNames).toHaveLength(12);
  expect(PYTHON_MCP_TOOL_COUNT).toBe(pythonNames.length);
  expect(PUBLIC_SITE_STATS.pythonMcpTools).toBe(pythonNames.length);
});

test("contract tools and differences match the current native and Python inventories", () => {
  const contract = readFileSync(
    new URL("../contracts/engine-ui-contract.md", import.meta.url), "utf8",
  );
  const canonical = contract.split("Canonical BrainBar tools:")[1].split("\n\nPython")[0];
  const names = (text: string) => [...text.matchAll(/`(brain_[a-z_]+)`/g)]
    .map((match) => match[1]).sort();
  expect(names(canonical)).toEqual(routerNames);
  expect(contract).toContain("16 canonical definitions");
  expect(contract).toContain("library-only surface with 12 canonical definitions");
  expect(names(contract.match(/- BrainBar-only: ([^\n]+)/)![1]))
    .toEqual(routerNames.filter((name) => !pythonNames.includes(name)));
  expect(names(contract.match(/- Python-only: ([^\n]+)/)![1]))
    .toEqual(pythonNames.filter((name) => !routerNames.includes(name)));
});
