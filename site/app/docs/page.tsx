"use client";

import { motion } from "framer-motion";
import Image from "next/image";
import Link from "next/link";

const fade = (delay: number) => ({
  initial: { opacity: 1, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, ease: "easeOut" as const, delay },
});

const sections = [
  {
    title: "Quick Start",
    desc: "Install, configure MCP, and start indexing in 3 commands.",
    href: "https://etanhey.github.io/brainlayer/quickstart/",
    icon: (
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
      </svg>
    ),
  },
  {
    title: "Configuration",
    desc: "MCP server setup, Claude Code integration, environment variables.",
    href: "https://etanhey.github.io/brainlayer/configuration/",
    icon: (
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <circle cx="12" cy="12" r="3" />
        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
      </svg>
    ),
  },
  {
    title: "MCP Tools",
    desc: "Full reference for brain_search, brain_store, brain_recall, and 5 more tools.",
    href: "https://etanhey.github.io/brainlayer/mcp-tools/",
    icon: (
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
      </svg>
    ),
  },
  {
    title: "Architecture",
    desc: "SQLite + sqlite-vec, hybrid search pipeline, knowledge graph design.",
    href: "https://etanhey.github.io/brainlayer/architecture/",
    icon: (
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </svg>
    ),
  },
  {
    title: "Enrichment",
    desc: "Auto-enrich chunks with entity extraction, tagging, and knowledge graph links.",
    href: "https://etanhey.github.io/brainlayer/enrichment/",
    icon: (
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <path d="M12 2L2 7l10 5 10-5-10-5z" />
        <path d="M2 17l10 5 10-5" />
        <path d="M2 12l10 5 10-5" />
      </svg>
    ),
  },
  {
    title: "Knowledge Graph",
    desc: "Entity extraction, relation mapping, GliNER integration for EN+HE.",
    href: "https://etanhey.github.io/brainlayer/kg-spec/",
    icon: (
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <circle cx="12" cy="5" r="3" />
        <circle cx="5" cy="19" r="3" />
        <circle cx="19" cy="19" r="3" />
        <line x1="12" y1="8" x2="5" y2="16" />
        <line x1="12" y1="8" x2="19" y2="16" />
      </svg>
    ),
  },
];

const guides = [
  {
    title: "Data Locations",
    href: "https://etanhey.github.io/brainlayer/data-locations/",
  },
  {
    title: "Enrichment Runbook",
    href: "https://etanhey.github.io/brainlayer/enrichment-runbook/",
  },
  {
    title: "Local Models Guide",
    href: "https://etanhey.github.io/brainlayer/local-models-guide/",
  },
  {
    title: "MCP Config",
    href: "https://etanhey.github.io/brainlayer/mcp-config/",
  },
];

export default function DocsPage() {
  return (
    <>
      {/* Nav */}
      <nav className="fixed top-0 left-0 right-0 z-50 py-4 backdrop-blur-xl border-b border-border bg-bg/80">
        <div className="mx-auto flex max-w-[960px] items-center justify-between px-6">
          <Link
            href="/"
            className="flex items-center gap-2.5 text-text opacity-90 transition-opacity hover:opacity-100"
          >
            <Image
              src="/logos/brainlayer.svg"
              alt=""
              width={24}
              height={24}
              className="hue-rotate-[120deg] saturate-[0.65] brightness-110"
            />
            <span className="font-sans text-base font-semibold tracking-tight">
              BrainLayer
            </span>
          </Link>
          <div className="flex items-center gap-6">
            <Link
              href="/"
              className="text-sm text-text-secondary transition-colors hover:text-text"
            >
              Home
            </Link>
            <span className="text-sm text-text font-medium">Docs</span>
            <a
              href="https://github.com/EtanHey/brainlayer"
              className="flex items-center gap-1.5 text-sm text-text-secondary transition-colors hover:text-text"
            >
              GitHub
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className="ml-0.5 opacity-50"
              >
                <path d="M3 9L9 3M9 3H4.5M9 3v4.5" />
              </svg>
            </a>
          </div>
        </div>
      </nav>

      <main className="pt-32 pb-24">
        <div className="mx-auto max-w-[960px] px-6">
          {/* Header */}
          <motion.div className="mb-16" {...fade(0)}>
            <div className="mb-3 text-[11px] font-medium uppercase tracking-[0.12em] text-accent">
              Documentation
            </div>
            <h1 className="font-display text-[clamp(32px,5vw,48px)] font-bold tracking-[-0.03em] leading-[1.1] mb-4">
              Learn BrainLayer
            </h1>
            <p className="text-lg text-text-secondary font-light max-w-[560px]">
              Everything you need to set up persistent memory for your AI
              agents. From installation to advanced knowledge graph features.
            </p>
          </motion.div>

          {/* Main sections grid */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-16"
            {...fade(0.15)}
          >
            {sections.map((section) => (
              <a
                key={section.title}
                href={section.href}
                className="group flex flex-col gap-3 rounded-xl border border-border bg-bg-card p-6 transition-all hover:border-border-hover hover:bg-bg-elevated"
              >
                <div className="flex items-center gap-3">
                  <div className="text-accent">{section.icon}</div>
                  <h3 className="font-sans text-base font-semibold tracking-tight group-hover:text-accent transition-colors">
                    {section.title}
                  </h3>
                </div>
                <p className="text-sm text-text-secondary font-light leading-relaxed">
                  {section.desc}
                </p>
              </a>
            ))}
          </motion.div>

          {/* Guides */}
          <motion.div {...fade(0.3)}>
            <h2 className="font-display text-xl font-semibold tracking-tight mb-6">
              Guides
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {guides.map((guide) => (
                <a
                  key={guide.title}
                  href={guide.href}
                  className="group flex items-center gap-3 rounded-lg border border-border px-5 py-4 transition-all hover:border-border-hover hover:bg-bg-elevated"
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    className="text-text-dim group-hover:text-accent transition-colors shrink-0"
                  >
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <path d="M14 2v6h6" />
                    <line x1="16" y1="13" x2="8" y2="13" />
                    <line x1="16" y1="17" x2="8" y2="17" />
                  </svg>
                  <span className="text-sm font-medium text-text-secondary group-hover:text-text transition-colors">
                    {guide.title}
                  </span>
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    className="ml-auto text-text-dim group-hover:text-accent transition-all group-hover:translate-x-0.5"
                  >
                    <path d="M5 12h14M12 5l7 7-7 7" />
                  </svg>
                </a>
              ))}
            </div>
          </motion.div>

          {/* ADRs */}
          <motion.div className="mt-12" {...fade(0.4)}>
            <h2 className="font-display text-xl font-semibold tracking-tight mb-6">
              Architecture Decisions
            </h2>
            <div className="space-y-2">
              <a
                href="https://etanhey.github.io/brainlayer/adr/0001-sqlite-vec-over-dedicated-vector-db/"
                className="group flex items-baseline gap-3 rounded-lg px-5 py-3 transition-colors hover:bg-bg-elevated"
              >
                <span className="font-mono text-xs text-accent shrink-0">
                  ADR-0001
                </span>
                <span className="text-sm text-text-secondary group-hover:text-text transition-colors">
                  sqlite-vec over dedicated vector DB
                </span>
              </a>
              <a
                href="https://etanhey.github.io/brainlayer/adr/0002-reciprocal-rank-fusion-for-hybrid-search/"
                className="group flex items-baseline gap-3 rounded-lg px-5 py-3 transition-colors hover:bg-bg-elevated"
              >
                <span className="font-mono text-xs text-accent shrink-0">
                  ADR-0002
                </span>
                <span className="text-sm text-text-secondary group-hover:text-text transition-colors">
                  Reciprocal Rank Fusion for hybrid search
                </span>
              </a>
            </div>
          </motion.div>

          {/* Full docs link */}
          <motion.div className="mt-16 text-center" {...fade(0.5)}>
            <a
              href="https://etanhey.github.io/brainlayer"
              className="inline-flex items-center gap-2 rounded-full border border-border px-6 py-3 text-sm font-medium text-text-secondary transition-all hover:scale-[1.03] hover:border-border-hover hover:text-text active:scale-[0.98]"
            >
              View full documentation
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3" />
              </svg>
            </a>
          </motion.div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-8">
        <div className="mx-auto flex max-w-[960px] flex-col items-center justify-between gap-3 px-6 sm:flex-row">
          <div className="text-[13px] font-light text-text-dim">
            Built by{" "}
            <a
              href="https://etanheyman.com"
              className="text-text-secondary transition-colors hover:text-accent"
            >
              Etan Heyman
            </a>
          </div>
          <div className="flex gap-5">
            <a
              href="https://github.com/EtanHey/brainlayer"
              className="text-[13px] text-text-dim transition-colors hover:text-text-secondary"
            >
              GitHub
            </a>
            <a
              href="https://pypi.org/project/brainlayer/"
              className="text-[13px] text-text-dim transition-colors hover:text-text-secondary"
            >
              PyPI
            </a>
          </div>
        </div>
      </footer>
    </>
  );
}
