"use client";

import { motion } from "framer-motion";

export function ProblemSolution() {
  return (
    <section className="py-24">
      <div className="mx-auto grid max-w-[960px] grid-cols-1 gap-6 px-6 md:grid-cols-2">
        <motion.div
          className="flex flex-col rounded-xl border border-border border-t-2 border-t-red bg-bg-card p-8 transition-colors hover:border-border-hover"
          initial={{ opacity: 1, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-3.5 text-[11px] font-medium uppercase tracking-[0.1em] text-red">
            The problem
          </div>
          <h2 className="mb-3.5 font-sans text-xl font-semibold tracking-tight">
            Your AI has amnesia
          </h2>
          <p className="flex-1 text-[15px] leading-relaxed font-light text-text-secondary">
            Every session starts from zero. Your agent doesn&apos;t remember the
            architecture decision from last week, the debugging approach that
            worked, or that you prefer tabs over spaces.
            <br />
            <br />
            You explain the same context{" "}
            <strong className="font-medium text-text">dozens of times</strong>.
            It suggests patterns you&apos;ve already rejected.
          </p>
        </motion.div>

        <motion.div
          className="flex flex-col rounded-xl border border-border border-t-2 border-t-teal bg-bg-card p-8 transition-colors hover:border-border-hover"
          initial={{ opacity: 1, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="mb-3.5 text-[11px] font-medium uppercase tracking-[0.1em] text-teal">
            The fix
          </div>
          <h2 className="mb-3.5 font-sans text-xl font-semibold tracking-tight">
            Install. Index. Search.
          </h2>
          <p className="flex-1 text-[15px] leading-relaxed font-light text-text-secondary">
            BrainLayer indexes your AI conversations into{" "}
            <strong className="font-medium text-text">
              284K searchable chunks
            </strong>
            . Hybrid search (semantic + keyword + knowledge graph) finds the
            right memory in{" "}
            <strong className="font-medium text-text">&lt;50ms</strong>.
            <br />
            <br />
            SessionStart hooks auto-inject relevant context before you even ask.
            Everything stays on your machine. One SQLite file.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
