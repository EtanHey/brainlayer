export const BRAINBAR_MCP_TOOL_GROUPS = [
  {
    label: "Core - what you use daily",
    tools: [
      {
        name: "brain_search",
        desc: "Hybrid semantic + keyword + KG search with compact formatted output",
      },
      {
        name: "brain_store",
        desc: "Persist decisions, learnings, corrections, and capture notes",
      },
      {
        name: "brain_recall",
        desc: "Session-aware context, history, plans, and knowledge-base stats",
      },
      {
        name: "brain_expand",
        desc: "Drill into one search hit with surrounding context",
      },
    ],
  },
  {
    label: "Knowledge and lifecycle",
    tools: [
      { name: "brain_entity", desc: "Knowledge graph entity lookup" },
      {
        name: "brain_get_person",
        desc: "Look up a person with graph relations and linked memories",
      },
      {
        name: "brain_digest",
        desc: "Deep-ingest large content and extract entities and relations",
      },
      {
        name: "brain_update",
        desc: "Update chunk importance and tags by chunk ID",
      },
      {
        name: "brain_tags",
        desc: "List unique tags with counts and optional filtering",
      },
      {
        name: "brain_supersede",
        desc: "Replace a chunk with a newer version, preserving history",
      },
      {
        name: "brain_archive",
        desc: "Archive a chunk while preserving its audit trail",
      },
      {
        name: "brain_enrich",
        desc: "Backfill summaries and enrichment metadata",
      },
    ],
  },
  {
    label: "Agent bus",
    tools: [
      {
        name: "brain_subscribe",
        desc: "Subscribe an agent to notifications for matching tags",
      },
      {
        name: "brain_unsubscribe",
        desc: "Remove some or all tag subscriptions for an agent",
      },
      {
        name: "brain_ack",
        desc: "Acknowledge messages processed by an agent",
      },
    ],
  },
  {
    label: "Operator",
    tools: [
      {
        name: "brain_backup_vacuum_into",
        desc: "Create a consistent SQLite backup snapshot",
      },
      {
        name: "brain_maintenance_rebuild_trigram",
        desc: "Rebuild the trigram search index in lock-aware batches",
      },
    ],
  },
] as const;

export const BRAINBAR_MCP_TOOL_COUNT = BRAINBAR_MCP_TOOL_GROUPS.reduce(
  (count, group) => count + group.tools.length,
  0,
);

export const PYTHON_MCP_TOOL_COUNT = 13;
