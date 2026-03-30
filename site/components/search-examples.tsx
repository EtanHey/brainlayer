"use client";

import { motion } from "framer-motion";

const examples = [
  {
    title: "Finding the BrainBar rollout details",
    command: 'brain_search(query="brainbar quick capture")',
    caption:
      "Real MCP formatter output: Unicode box drawing, ranked hits, compact tags.",
    output: `┌─ brain_search: "brainbar quick capture" ─ 2 results
│
├─ [1] agent-a34f46  score:0.93  imp: 9  2026-03-30
│  brainlayer       │ BrainBar now enforces a single running instance and exits immediately i…
│  tags: brainbar, single-instance, macos
│
├─ [2] agent-c82d91  score:0.88  imp: 8  2026-03-29
│  brainlayer       │ Quick capture uses the shared Unicode formatter, so search results from…
│  tags: quick-capture, formatting
│
└─`,
  },
  {
    title: "Compact output with no tag bloat",
    command: 'brain_search(query="formatted output tags")',
    caption:
      "When a memory has no tags, the formatter omits the line instead of printing empty noise.",
    output: `┌─ brain_search: "formatted output tags" ─ 1 result
│
├─ [1] agent-e12c44  score:0.91  imp: 7  2026-03-30
│  brainlayer       │ Search formatting omits empty tag lines, so the terminal output stays c…
│
└─`,
  },
];

export function SearchExamples() {
  return (
    <section id="examples" className="py-24">
      <div className="mx-auto max-w-[1120px] px-6">
        <div className="mb-3 text-center text-[11px] font-medium uppercase tracking-[0.12em] text-accent">
          Real Output
        </div>
        <h2 className="mx-auto mb-4 max-w-[680px] text-center font-display text-[clamp(26px,3.8vw,40px)] font-semibold leading-tight tracking-tight text-balance">
          `brain_search` now looks like a tool you can actually trust
        </h2>
        <p className="mx-auto mb-14 max-w-[720px] text-center text-[15px] leading-relaxed font-light text-text-secondary">
          These examples come from the current formatter in
          `src/brainlayer/mcp/_format.py` and the matching BrainBar Swift port.
          Same ranked results, same truncation, same box-drawing output.
        </p>

        <div className="grid gap-6 lg:grid-cols-2">
          {examples.map((example, index) => (
            <motion.article
              key={example.title}
              className="rounded-2xl border border-border bg-bg-card p-6"
              initial={{ opacity: 1, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.45, delay: index * 0.08 }}
            >
              <div className="mb-4 flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.1em] text-text-dim">
                <span className="inline-block h-2 w-2 rounded-full bg-accent" />
                Usage Example
              </div>
              <h3 className="mb-2 font-sans text-lg font-semibold tracking-tight text-text">
                {example.title}
              </h3>
              <p className="mb-4 text-sm leading-relaxed font-light text-text-secondary">
                {example.caption}
              </p>
              <div className="mb-3 rounded-xl border border-border bg-bg-elevated px-4 py-3 font-mono text-[12px] text-accent-bright">
                {example.command}
              </div>
              <pre className="overflow-x-auto rounded-xl border border-border bg-[#070709] px-4 py-4 font-mono text-[12px] leading-6 text-[#e7ddd3]">
                <code>{example.output}</code>
              </pre>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
