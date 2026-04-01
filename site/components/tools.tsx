"use client";

import { motion } from "framer-motion";

const coreTools = [
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
    desc: "Unified recall entrypoint for search, entity, and session-aware lookup",
  },
];

const advancedTools = [
  { name: "brain_entity", desc: "Knowledge graph entity lookup" },
  {
    name: "brain_expand",
    desc: "Drill into one search hit with surrounding context",
  },
  {
    name: "brain_digest",
    desc: "Deep-ingest large content and extract entities, actions, and relations",
  },
  {
    name: "brain_update",
    desc: "Update chunk importance and tags by chunk ID",
  },
  {
    name: "brain_tags",
    desc: "List unique tags with counts, filter by prefix",
  },
  {
    name: "brain_get_person",
    desc: "Look up a person entity with all known relations",
  },
];

const lifecycleTools = [
  {
    name: "brain_supersede",
    desc: "Replace a chunk with a newer version, preserving history",
  },
  {
    name: "brain_archive",
    desc: "Soft-delete a chunk with timestamp for audit trail",
  },
  {
    name: "brain_enrich",
    desc: "Trigger deep enrichment on a stored chunk via Groq or Gemini",
  },
];

function ToolItem({ name, desc }: { name: string; desc: string }) {
  return (
    <motion.div
      className="flex items-baseline gap-5 rounded-lg px-4 py-3 transition-colors hover:bg-bg-elevated"
      initial={{ opacity: 1, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4 }}
    >
      <span className="min-w-[150px] shrink-0 font-mono text-[13px] font-medium text-accent">
        {name}
      </span>
      <span className="text-sm font-light text-text-secondary">{desc}</span>
    </motion.div>
  );
}

export function Tools() {
  return (
    <section id="tools" className="py-24">
      <div className="mx-auto max-w-[960px] px-6">
        <div className="mb-3 text-center text-[11px] font-medium uppercase tracking-[0.12em] text-accent">
          MCP tools
        </div>
        <h2 className="mb-14 text-center font-display text-[clamp(26px,3.5vw,36px)] font-semibold leading-tight tracking-tight text-balance">
          Twelve working tools. One memory layer.
        </h2>

        <div className="mx-auto mb-12 max-w-[640px]">
          <div className="mb-4 pl-1 text-xs font-medium uppercase tracking-[0.1em] text-text-dim">
            Core - what you use daily
          </div>
          {coreTools.map((tool) => (
            <ToolItem key={tool.name} {...tool} />
          ))}
        </div>

        <div className="mx-auto h-px max-w-[640px] bg-border" />

        <div className="mx-auto mt-2 max-w-[640px]">
          <div className="mb-4 mt-6 pl-1 text-xs font-medium uppercase tracking-[0.1em] text-text-dim">
            Advanced
          </div>
          {advancedTools.map((tool) => (
            <ToolItem key={tool.name} {...tool} />
          ))}
        </div>

        <div className="mx-auto h-px max-w-[640px] bg-border" />

        <div className="mx-auto mt-2 max-w-[640px]">
          <div className="mb-4 mt-6 pl-1 text-xs font-medium uppercase tracking-[0.1em] text-text-dim">
            Lifecycle
          </div>
          {lifecycleTools.map((tool) => (
            <ToolItem key={tool.name} {...tool} />
          ))}
        </div>
      </div>
    </section>
  );
}
