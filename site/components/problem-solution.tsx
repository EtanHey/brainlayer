"use client";

import { motion } from "framer-motion";

export function ProblemSolution() {
  return (
    <section id="why" className="py-24">
      <div className="mx-auto max-w-[1120px] px-6">
        <div className="mb-3 text-center text-[11px] font-medium uppercase tracking-[0.12em] text-accent">
          The Problem
        </div>
        <h2 className="mx-auto mb-4 max-w-[720px] text-center font-display text-[clamp(26px,3.8vw,40px)] font-semibold leading-tight tracking-tight text-balance">
          Your AI forgets everything between sessions
        </h2>
        <p className="mx-auto mb-12 max-w-[760px] text-center text-[15px] leading-relaxed font-light text-text-secondary">
          Every architecture decision, every debugging session, every correction
          you gave it &mdash; gone. BrainLayer gives any MCP agent persistent
          memory backed by semantic search and a knowledge graph.
        </p>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <motion.div
            className="flex flex-col rounded-xl border border-border border-t-2 border-t-accent bg-bg-card p-8 transition-colors hover:border-border-hover"
            initial={{ opacity: 1, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <div className="mb-3.5 text-[11px] font-medium uppercase tracking-[0.1em] text-accent">
              Semantic search
            </div>
            <h2 className="mb-3.5 font-sans text-xl font-semibold tracking-tight text-balance">
              Hybrid vector + keyword search with RRF
            </h2>
            <p className="flex-1 text-[15px] leading-relaxed font-light text-text-secondary">
              <code className="text-accent text-xs">brain_search</code> combines
              bge-large embeddings with FTS5 keyword matching via Reciprocal
              Rank Fusion. One query searches across every conversation you have
              ever had. Sub-50ms on 300K+ chunks.
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
              Knowledge graph
            </div>
            <h2 className="mb-3.5 font-sans text-xl font-semibold tracking-tight text-balance">
              Entities and relations that grow over time
            </h2>
            <p className="flex-1 text-[15px] leading-relaxed font-light text-text-secondary">
              <code className="text-accent text-xs">brain_digest</code> extracts
              entities, relations, and action items from raw content.
              <code className="text-accent text-xs"> brain_entity</code> looks
              up any entity in the graph with evidence and connections.
            </p>
          </motion.div>

          <motion.div
            className="flex flex-col rounded-xl border border-border border-t-2 border-t-accent bg-bg-card p-8 transition-colors hover:border-border-hover"
            initial={{ opacity: 1, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <div className="mb-3.5 text-[11px] font-medium uppercase tracking-[0.1em] text-accent">
              Persistent memory
            </div>
            <h2 className="mb-3.5 font-sans text-xl font-semibold tracking-tight text-balance">
              Decisions, learnings, corrections &mdash; stored forever
            </h2>
            <p className="flex-1 text-[15px] leading-relaxed font-light text-text-secondary">
              <code className="text-accent text-xs">brain_store</code> persists
              any memory with auto-type detection, auto-importance scoring, and
              per-agent scoping. Chunk lifecycle management (supersede, archive)
              keeps knowledge current without losing history.
            </p>
          </motion.div>

          <motion.div
            className="flex flex-col rounded-xl border border-dashed border-text-dim/30 bg-bg-elevated/50 p-8 transition-colors hover:border-text-dim/50"
            initial={{ opacity: 1, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <div className="mb-1 text-[10px] font-medium uppercase tracking-[0.12em] text-text-dim">
              Companion App &mdash; macOS
            </div>
            <h2 className="mb-3.5 font-sans text-xl font-semibold tracking-tight text-balance">
              BrainBar
            </h2>
            <p className="flex-1 text-[15px] leading-relaxed font-light text-text-secondary">
              Optional 209KB native Swift menu bar app. Quick capture, live
              dashboard, knowledge graph viewer &mdash; all over a Unix socket.
            </p>
            <p className="mt-3 text-[12px] font-light text-text-dim">
              Requires the BrainLayer MCP server.
            </p>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
