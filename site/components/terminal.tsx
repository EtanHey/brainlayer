"use client";

import { useEffect, useState, useRef } from "react";
import { motion, useInView } from "framer-motion";

interface TerminalLine {
  type:
    | "prompt"
    | "sys"
    | "body"
    | "border-start"
    | "border"
    | "border-meta"
    | "border-str"
    | "border-end"
    | "cursor";
  content: string;
  gap?: boolean;
  /** ms to wait before showing this line */
  delay?: number;
}

const lines: TerminalLine[] = [
  {
    type: "prompt",
    content: "What database architecture does BrainLayer use?",
    delay: 0,
  },
  {
    type: "sys",
    content: "[BrainLayer] SessionStart hook \u00b7 4 chunks injected (8ms)",
    gap: true,
    delay: 1200,
  },
  {
    type: "body",
    content: "Let me search our architecture decisions.",
    gap: true,
    delay: 1800,
  },
  {
    type: "border-start",
    content:
      '\u250c\u2500 brain_search(query="BrainLayer architecture hybrid search pipeline")',
    gap: true,
    delay: 2400,
  },
  { type: "border", content: "\u2502", delay: 2500 },
  {
    type: "border-meta",
    content: "\u2502  chunks: 3  latency: 9ms  top: 0.94",
    delay: 2600,
  },
  { type: "border", content: "\u2502", delay: 2700 },
  {
    type: "border-str",
    content:
      '\u2502  "Claude Code / Cursor / Zed \u2192 MCP \u2192 BrainLayer Server',
    delay: 2800,
  },
  {
    type: "border-str",
    content: "\u2502   \u2192 Hybrid Search (semantic + keyword via RRF)",
    delay: 2900,
  },
  {
    type: "border-str",
    content: "\u2502   \u2192 SQLite + sqlite-vec, single .db file",
    delay: 3000,
  },
  {
    type: "border-str",
    content: '\u2502   \u2192 Knowledge Graph (entities + relations)"',
    delay: 3100,
  },
  { type: "border", content: "\u2502", delay: 3200 },
  {
    type: "border-end",
    content:
      "\u2514\u2500 agent-a34f466 \u00b7 importance: 8 \u00b7 tags: architecture",
    delay: 3300,
  },
  {
    type: "body",
    content: "BrainLayer uses a single SQLite file with sqlite-vec for",
    gap: true,
    delay: 3800,
  },
  {
    type: "body",
    content: "vector storage. Search fuses semantic, FTS5 keyword, and",
    delay: 3900,
  },
  {
    type: "body",
    content: "knowledge graph signals via Reciprocal Rank Fusion.",
    delay: 4000,
  },
];

function useTypingEffect(
  text: string,
  isActive: boolean,
  speed: number = 35,
): string {
  const [displayed, setDisplayed] = useState("");

  useEffect(() => {
    if (!isActive) {
      setDisplayed("");
      return;
    }
    setDisplayed("");
    let i = 0;
    const interval = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) clearInterval(interval);
    }, speed);
    return () => clearInterval(interval);
  }, [text, isActive, speed]);

  return displayed;
}

function PromptLine({
  content,
  isActive,
}: {
  content: string;
  isActive: boolean;
}) {
  const typed = useTypingEffect(content, isActive, 30);
  const showCursor = isActive && typed.length < content.length;

  return (
    <span className="block">
      <span className="text-[#6ec1e4]">{"\u276f"}</span>{" "}
      <span className="text-text">
        {typed}
        {showCursor && (
          <span className="inline-block w-[7px] h-[15px] bg-text/70 animate-pulse ml-px translate-y-[2px]" />
        )}
      </span>
    </span>
  );
}

function RenderLine({ line }: { line: TerminalLine }) {
  const { type, content, gap } = line;
  const gapClass = gap ? "mt-3" : "";

  if (type === "prompt") {
    // Handled separately by PromptLine
    return null;
  }

  if (type === "sys") {
    return (
      <span className={`block ${gapClass} italic text-[#78787f]`}>
        {content}
      </span>
    );
  }

  if (type === "body") {
    return (
      <span className={`block ${gapClass} text-text-secondary`}>{content}</span>
    );
  }

  if (type === "border-start") {
    const match = content.match(/^(\u250c\u2500)\s*(brain_search)\((.+)\)$/);
    if (match) {
      return (
        <span className={`block ${gapClass}`}>
          <span className="text-[#333338]">{match[1]}</span>{" "}
          <span className="font-medium text-accent">{match[2]}</span>
          <span className="text-text-secondary">({match[3]})</span>
        </span>
      );
    }
    return <span className={`block ${gapClass}`}>{content}</span>;
  }

  if (type === "border") {
    return <span className="block text-[#333338]">{content}</span>;
  }

  if (type === "border-meta") {
    return (
      <span className="block">
        <span className="text-[#333338]">{"\u2502"}</span>
        <span className="text-[#4a4a52]">
          {"  "}chunks: <span className="text-teal">3</span>
          {"  "}latency: <span className="text-teal">9ms</span>
          {"  "}top: <span className="text-teal">0.94</span>
        </span>
      </span>
    );
  }

  if (type === "border-str") {
    const border = content.substring(0, 1);
    const text = content.substring(1);
    return (
      <span className="block">
        <span className="text-[#333338]">{border}</span>
        <span className="text-accent-bright">{text}</span>
      </span>
    );
  }

  if (type === "border-end") {
    const border = content.substring(0, 2);
    const text = content.substring(2);
    return (
      <span className="block">
        <span className="text-[#333338]">{border}</span>
        <span className="text-[#4a4a52]">{text}</span>
      </span>
    );
  }

  return <span className="block text-text-secondary">{content}</span>;
}

export function Terminal() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  const [visibleCount, setVisibleCount] = useState(0);
  const [promptDone, setPromptDone] = useState(false);
  const timeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    if (!isInView) return;

    // Clear any existing timeouts
    timeoutsRef.current.forEach(clearTimeout);
    timeoutsRef.current = [];

    // Show prompt immediately (typing animation handles its reveal)
    setVisibleCount(1);

    // Mark prompt as done typing after a delay matching the text length
    const promptLength = lines[0].content.length;
    const promptTimeout = setTimeout(
      () => {
        setPromptDone(true);
      },
      promptLength * 30 + 200,
    );
    timeoutsRef.current.push(promptTimeout);

    // Schedule remaining lines
    for (let i = 1; i < lines.length; i++) {
      const baseDelay = (lines[i].delay ?? 0) + promptLength * 30;
      const t = setTimeout(() => {
        setVisibleCount((prev) => Math.max(prev, i + 1));
      }, baseDelay);
      timeoutsRef.current.push(t);
    }

    return () => {
      timeoutsRef.current.forEach(clearTimeout);
    };
  }, [isInView]);

  return (
    <section className="pb-24">
      <div className="mx-auto max-w-[1200px] px-6">
        <motion.div
          ref={ref}
          className="relative mx-auto max-w-[820px] overflow-hidden rounded-2xl border border-white/[0.06] bg-[#0c0c0e] shadow-[0_0_80px_rgba(212,149,106,0.04)]"
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-40px" }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          {/* Gradient fade at bottom */}
          <div className="pointer-events-none absolute right-0 bottom-0 left-0 h-16 bg-gradient-to-t from-bg to-transparent z-10" />

          {/* Title bar */}
          <div className="flex items-center gap-[7px] border-b border-white/[0.05] bg-white/[0.03] px-[18px] py-3.5">
            <div className="h-[11px] w-[11px] rounded-full bg-[#ff5f57]" />
            <div className="h-[11px] w-[11px] rounded-full bg-[#febc2e]" />
            <div className="h-[11px] w-[11px] rounded-full bg-[#28c840]" />
            <span className="ml-2.5 font-mono text-xs text-text-dim">
              claude ~ myproject
            </span>
          </div>

          {/* Body */}
          <div className="px-[22px] pt-5 pb-14 font-mono text-[13px] leading-[1.85] min-h-[380px]">
            {/* Prompt line with typing animation */}
            {visibleCount >= 1 && (
              <PromptLine content={lines[0].content} isActive={isInView} />
            )}

            {/* Rest of lines revealed progressively */}
            {lines.slice(1).map((line, i) => {
              if (i + 1 >= visibleCount) return null;
              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.15 }}
                >
                  <RenderLine line={line} />
                </motion.div>
              );
            })}

            {/* Blinking cursor at the end when prompt is typing and no other lines shown */}
            {promptDone && visibleCount <= 1 && (
              <span className="block mt-3">
                <span className="inline-block w-[7px] h-[15px] bg-text/40 animate-pulse" />
              </span>
            )}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
