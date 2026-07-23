"use client";

import { motion } from "framer-motion";
import { PUBLIC_SITE_STATS } from "@/lib/public-stats";
import { BRAINBAR_MCP_TOOL_GROUPS } from "@/lib/public-tools";

function ToolItem({ name, desc }: { name: string; desc: string }) {
  return (
    <motion.div
      className="flex flex-col items-start gap-1.5 rounded-lg px-4 py-3 transition-colors hover:bg-bg-elevated sm:flex-row sm:items-baseline sm:gap-5"
      initial={{ opacity: 1, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4 }}
    >
      <span className="shrink-0 font-mono text-[13px] font-medium text-accent sm:min-w-[150px]">
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
          {PUBLIC_SITE_STATS.brainBarMcpTools}-tool live surface. One memory
          layer.
        </h2>

        {BRAINBAR_MCP_TOOL_GROUPS.map((group, index) => (
          <div key={group.label}>
            {index > 0 && (
              <div className="mx-auto h-px max-w-[640px] bg-border" />
            )}
            <div className="mx-auto mb-2 mt-6 max-w-[640px]">
              <div className="mb-4 pl-1 text-xs font-medium uppercase tracking-[0.1em] text-text-dim">
                {group.label}
              </div>
              {group.tools.map((tool) => (
                <ToolItem key={tool.name} {...tool} />
              ))}
            </div>
          </div>
        ))}

        <p className="mx-auto mt-8 max-w-[640px] text-[12px] font-light leading-relaxed text-text-dim">
          The secondary Python transport exposes{" "}
          {PUBLIC_SITE_STATS.pythonMcpTools} tools, including the Python-only{" "}
          <code className="text-accent">brain_resume</code>.
        </p>
      </div>
    </section>
  );
}
