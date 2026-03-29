"use client";

import { useEffect, useState, useRef } from "react";
import { motion, useInView } from "framer-motion";

interface Line {
  type:
    | "prompt"
    | "claude"
    | "tool"
    | "output"
    | "output-first"
    | "result"
    | "hr"
    | "status";
  text: string;
  gap?: boolean;
  delay: number;
}

// Matches real Claude Code output format exactly
// Verified by reading own terminal via cmux read_screen
const lines: Line[] = [
  {
    type: "prompt",
    text: "What database architecture does BrainLayer use?",
    delay: 0,
  },
  {
    type: "claude",
    text: "I'll search our past architecture decisions.",
    gap: true,
    delay: 800,
  },
  {
    type: "tool",
    text: 'brain_search (MCP)(query: "BrainLayer architecture")',
    gap: true,
    delay: 1200,
  },
  { type: "output-first", text: "[", delay: 1500 },
  { type: "output", text: "     {", delay: 1530 },
  { type: "output", text: '       "chunk_id": "agent-a34f466",', delay: 1560 },
  {
    type: "output",
    text: '       "summary": "BrainLayer MCP server architecture with',
    delay: 1590,
  },
  {
    type: "result",
    text: '                   hybrid search and knowledge graph.",',
    delay: 1620,
  },
  {
    type: "output",
    text: '       "content": "Claude Code / Cursor / Zed -> MCP ->',
    delay: 1650,
  },
  {
    type: "result",
    text: "                   Hybrid Search (semantic + keyword RRF)",
    delay: 1680,
  },
  {
    type: "result",
    text: "                   -> SQLite + sqlite-vec, single .db file",
    delay: 1710,
  },
  {
    type: "result",
    text: '                   -> Knowledge Graph (entities + relations)",',
    delay: 1740,
  },
  { type: "output", text: '       "importance": 8,', delay: 1770 },
  {
    type: "output",
    text: '       "tags": ["architecture", "search", "sqlite"]',
    delay: 1800,
  },
  { type: "output", text: "     }", delay: 1830 },
  { type: "output", text: "   ]", delay: 1860 },
  {
    type: "claude",
    text: "BrainLayer uses a single SQLite file with sqlite-vec for",
    gap: true,
    delay: 2300,
  },
  {
    type: "claude",
    text: "vector storage. Search fuses semantic, FTS5 keyword, and",
    delay: 2400,
  },
  {
    type: "claude",
    text: "knowledge graph signals via Reciprocal Rank Fusion.",
    delay: 2500,
  },
];

function useTyping(text: string, active: boolean, speed = 30) {
  const [out, setOut] = useState("");
  useEffect(() => {
    if (!active) {
      setOut("");
      return;
    }
    setOut("");
    let i = 0;
    const iv = setInterval(() => {
      i++;
      setOut(text.slice(0, i));
      if (i >= text.length) clearInterval(iv);
    }, speed);
    return () => clearInterval(iv);
  }, [text, active, speed]);
  return out;
}

function PromptLine({ text, active }: { text: string; active: boolean }) {
  const typed = useTyping(text, active, 22);
  const showCursor = active && typed.length < text.length;
  return (
    <span className="block">
      <span className="text-[#6ec1e4]">{"❯"}</span>{" "}
      <span className="text-text">
        {typed}
        {showCursor && (
          <span className="inline-block w-[7px] h-[15px] bg-text/70 animate-pulse ml-px translate-y-[2px]" />
        )}
      </span>
    </span>
  );
}

function RenderLine({ line }: { line: Line }) {
  const g = line.gap ? "mt-3" : "";

  if (line.type === "prompt") return null; // handled separately

  if (line.type === "claude") {
    // First claude line in a group gets the ⏺ marker
    return (
      <span className={`block ${g} text-text-secondary`}>
        {line.gap && <span className="text-text-dim">{"⏺ "}</span>}
        {!line.gap && <span className="text-text-dim">{"  "}</span>}
        {line.text}
      </span>
    );
  }

  if (line.type === "tool") {
    const match = line.text.match(/^(\S+)\s+\(MCP\)\((.+)\)$/);
    if (match) {
      return (
        <span className={`block ${g}`}>
          <span className="text-text-dim">{"⏺ "}</span>
          <span className="font-medium text-accent">{match[1]}</span>
          <span className="text-text-dim">{" (MCP)"}</span>
          <span className="text-text-secondary">({match[2]})</span>
        </span>
      );
    }
    return (
      <span className={`block ${g} text-text-secondary`}>
        {"⏺ "}
        {line.text}
      </span>
    );
  }

  if (line.type === "output-first") {
    return (
      <span className={`block ${g} text-text-dim`}>
        {"  ⎿  "}
        {colorizeJson(line.text)}
      </span>
    );
  }

  if (line.type === "output") {
    return (
      <span className={`block ${g} text-text-dim`}>
        {"     "}
        {colorizeJson(line.text)}
      </span>
    );
  }

  if (line.type === "result") {
    return (
      <span className="block text-accent-bright">
        {"     "}
        {line.text}
      </span>
    );
  }

  return <span className={`block ${g} text-text-secondary`}>{line.text}</span>;
}

function colorizeJson(text: string) {
  // Split on quoted strings, colorize keys vs values
  return text.split(/("(?:[^"\\]|\\.)*")/).map((part, i) => {
    if (i % 2 === 1) {
      // Check if this quoted string is followed by a colon (it's a key)
      const afterQuote = text.indexOf(part) + part.length;
      const isKey = text[afterQuote] === ":";
      if (isKey) {
        return (
          <span key={i} className="text-[#8b9eb0]">
            {part}
          </span>
        );
      }
      return (
        <span key={i} className="text-accent-bright">
          {part}
        </span>
      );
    }
    // Numbers
    const withNums = part.replace(/\b(\d+)\b/g, "\x00$1\x01");
    if (withNums.includes("\x00")) {
      return withNums.split(/\x00|\x01/).map((seg, j) =>
        j % 2 === 1 ? (
          <span key={`${i}-${j}`} className="text-teal">
            {seg}
          </span>
        ) : (
          <span key={`${i}-${j}`}>{seg}</span>
        ),
      );
    }
    return <span key={i}>{part}</span>;
  });
}

export function Terminal() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  const [visibleCount, setVisibleCount] = useState(0);
  const timeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    if (!isInView) return;
    timeoutsRef.current.forEach(clearTimeout);
    timeoutsRef.current = [];
    setVisibleCount(1);

    const promptLen = lines[0].text.length;
    for (let i = 1; i < lines.length; i++) {
      const t = setTimeout(
        () => {
          setVisibleCount((prev) => Math.max(prev, i + 1));
        },
        (lines[i].delay ?? 0) + promptLen * 28,
      );
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
          initial={{ opacity: 1, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-40px" }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
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
          <div className="px-[22px] pt-5 pb-4 font-mono text-[13px] leading-[1.85] min-h-[380px] whitespace-pre-wrap">
            {visibleCount >= 1 && (
              <PromptLine text={lines[0].text} active={isInView} />
            )}
            {lines.slice(1).map((line, i) => {
              if (i + 1 >= visibleCount) return null;
              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 1 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.12 }}
                >
                  <RenderLine line={line} />
                </motion.div>
              );
            })}
          </div>

          {/* Status bar - matches real Claude Code */}
          <div className="px-[18px] font-mono text-text-dim">
            <div className="h-px bg-[#333338] mb-1" />
            <div className="flex items-center text-[#6ec1e4] py-0.5">
              <span>{"❯"}</span>
              <span className="ml-1 w-[7px] h-[13px] bg-text/30 animate-pulse" />
            </div>
            <div className="h-px bg-[#333338] mt-1 mb-1" />
            <div className="flex items-center justify-between pb-0.5 text-[10px]">
              <span>{"  ⎇ main | 🔧 7"}</span>
              <span>284,291 tokens</span>
            </div>
            <div className="flex items-center justify-between pb-2 text-[10px] opacity-60">
              <span>{"  🤖 Opus 4.6 (1M context)"}</span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
