"use client";

import { motion } from "framer-motion";

const without = [
  "Repeats the same mistakes every session",
  "Forgets architecture decisions overnight",
  "No context survives a restart or compaction",
  "Rediscovers bugs it already fixed last week",
  "Asks you the same clarifying questions again",
];

const withBL = [
  "Remembers every decision, learning, and correction",
  "Searches 294K+ knowledge chunks in under 50ms",
  "Knowledge graph connects entities across sessions",
  "Cross-session memory persists through restarts",
  "Recalls your preferences, patterns, and past work",
];

export function Comparison() {
  return (
    <section className="py-20">
      <div className="mx-auto max-w-[960px] px-6">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {/* WITHOUT */}
          <motion.div
            className="rounded-2xl border border-border bg-bg-card p-8"
            initial={{ opacity: 0, x: -16 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <div className="mb-6 text-[11px] font-medium uppercase tracking-[0.12em] text-red">
              Without BrainLayer
            </div>
            <ul className="space-y-4">
              {without.map((item) => (
                <li
                  key={item}
                  className="flex items-start gap-3 text-[15px] leading-relaxed text-text-secondary"
                >
                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-red/10 text-red text-xs font-bold">
                    ×
                  </span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </motion.div>

          {/* WITH */}
          <motion.div
            className="rounded-2xl border border-accent/20 bg-bg-card p-8"
            initial={{ opacity: 0, x: 16 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <div className="mb-6 text-[11px] font-medium uppercase tracking-[0.12em] text-accent">
              With BrainLayer
            </div>
            <ul className="space-y-4">
              {withBL.map((item) => (
                <li
                  key={item}
                  className="flex items-start gap-3 text-[15px] leading-relaxed text-text"
                >
                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent/10 text-accent text-xs font-bold">
                    ✓
                  </span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
