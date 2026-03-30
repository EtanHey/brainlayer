"use client";

import { motion } from "framer-motion";

export function ProblemSolution() {
  return (
    <section id="brainbar" className="py-24">
      <div className="mx-auto max-w-[1120px] px-6">
        <div className="mb-3 text-center text-[11px] font-medium uppercase tracking-[0.12em] text-accent">
          Current State
        </div>
        <h2 className="mx-auto mb-4 max-w-[720px] text-center font-display text-[clamp(26px,3.8vw,40px)] font-semibold leading-tight tracking-tight text-balance">
          The site now matches the product that actually ships
        </h2>
        <p className="mx-auto mb-12 max-w-[760px] text-center text-[15px] leading-relaxed font-light text-text-secondary">
          BrainLayer is no longer just a search box. The current release
          includes BrainBar capture and dashboard flows, compact formatter
          output, and a smaller, cleaner tool surface.
        </p>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <motion.div
          className="flex flex-col rounded-xl border border-border border-t-2 border-t-teal bg-bg-card p-8 transition-colors hover:border-border-hover"
          initial={{ opacity: 1, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-3.5 text-[11px] font-medium uppercase tracking-[0.1em] text-red">
            Keyboard-first capture
          </div>
          <h2 className="mb-3.5 font-sans text-xl font-semibold tracking-tight text-balance">
            F4 opens BrainBar quick capture
          </h2>
          <p className="flex-1 text-[15px] leading-relaxed font-light text-text-secondary">
            BrainBar listens for the default <strong className="font-medium text-text">F4</strong> hotkey and toggles a dedicated quick-capture panel
            for storing notes or jumping into search. The same search formatter
            powers both MCP and the macOS panel, so the output does not fork
            into a second UI language.
          </p>
        </motion.div>

        <motion.div
          className="flex flex-col rounded-xl border border-border border-t-2 border-t-accent bg-bg-card p-8 transition-colors hover:border-border-hover"
          initial={{ opacity: 1, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="mb-3.5 text-[11px] font-medium uppercase tracking-[0.1em] text-teal">
            Menu bar visibility
          </div>
          <h2 className="mb-3.5 font-sans text-xl font-semibold tracking-tight text-balance">
            Dashboard popover with live stats
          </h2>
          <p className="flex-1 text-[15px] leading-relaxed font-light text-text-secondary">
            The BrainBar status item opens a transient popover with chunk
            counts, enrichment progress, recent activity, daemon PID, RSS, and
            open socket metrics. It is a live operational view, not a static
            badge.
          </p>
        </motion.div>

        <motion.div
          className="flex flex-col rounded-xl border border-border border-t-2 border-t-red bg-bg-card p-8 transition-colors hover:border-border-hover"
          initial={{ opacity: 1, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="mb-3.5 text-[11px] font-medium uppercase tracking-[0.1em] text-red">
            Safer runtime
          </div>
          <h2 className="mb-3.5 font-sans text-xl font-semibold tracking-tight text-balance">
            Single-instance startup guard
          </h2>
          <p className="flex-1 text-[15px] leading-relaxed font-light text-text-secondary">
            BrainBar exits immediately if another copy already owns the bundle
            ID. That prevents duplicate menu-bar processes from competing for
            the same database and socket.
          </p>
        </motion.div>

        <motion.div
          className="flex flex-col rounded-xl border border-border border-t-2 border-t-accent bg-bg-card p-8 transition-colors hover:border-border-hover"
          initial={{ opacity: 1, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="mb-3.5 text-[11px] font-medium uppercase tracking-[0.1em] text-accent">
            Cleaner output
          </div>
          <h2 className="mb-3.5 font-sans text-xl font-semibold tracking-tight text-balance">
            Formatted Unicode output, no tag bloat
          </h2>
          <p className="flex-1 text-[15px] leading-relaxed font-light text-text-secondary">
            `brain_search` prints ranked results with box-drawing characters,
            scores, importance, dates, and only the tags that actually exist.
            Empty `tags:` placeholders are omitted to keep scans compact.
          </p>
        </motion.div>
        </div>
      </div>
    </section>
  );
}
