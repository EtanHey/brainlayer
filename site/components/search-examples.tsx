"use client";

import { useRef, useState, useEffect } from "react";
import { motion, useInView } from "framer-motion";

interface Line {
  type:
    | "prompt"
    | "claude"
    | "tool"
    | "output"
    | "output-first"
    | "result"
    | "tag-line";
  text: string;
  gap?: boolean;
  delay: number;
}

const lines: Line[] = [
  { type: "prompt", text: "Show me a compact search example.", delay: 0 },
  {
    type: "claude",
    text: "Running a search for formatted output handling.",
    gap: true,
    delay: 800,
  },
  {
    type: "tool",
    text: 'brain_search (MCP)(query: "formatted output tags")',
    gap: true,
    delay: 1200,
  },
  {
    type: "output-first",
    text: '┌─ brain_search: "formatted output tags" ─ 1 result',
    delay: 1500,
  },
  { type: "output", text: "│", delay: 1530 },
  {
    type: "output",
    text: "├─ [1] agent-e12c44  score:0.91  imp: 7  2026-03-30",
    delay: 1590,
  },
  {
    type: "result",
    text: "│  brainlayer       │ Search formatting omits empty tag lines,",
    delay: 1620,
  },
  {
    type: "result",
    text: "│                   │ so terminal output stays clean and compact…",
    delay: 1650,
  },
  { type: "output", text: "│", delay: 1680 },
  { type: "output", text: "└─", delay: 1710 },
  {
    type: "claude",
    text: "When a memory has no tags, the formatter omits the line",
    gap: true,
    delay: 2100,
  },
  {
    type: "claude",
    text: "instead of printing empty noise. Output stays compact.",
    delay: 2200,
  },
];

function useTyping(text: string, active: boolean, speed = 30) {
  const [out, setOut] = useState("");
  useEffect(() => {
    if (!active) {
      setOut("");
      return;
    }
    if (
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      setOut(text);
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

function colorize(text: string) {
  // Box drawing chars get warm copper
  if (text.match(/^[┌├└│]/) || text === "│") {
    // Colorize scores, importance, dates
    const parts = text.split(
      /(score:\d+\.\d+|imp:\s*\d+|\d{4}-\d{2}-\d{2}|agent-[a-f0-9]+)/g,
    );
    return parts.map((part, i) => {
      if (part.match(/^score:/))
        return (
          <span key={i} className="text-[#8B5CF6] font-medium">
            {part}
          </span>
        );
      if (part.match(/^imp:/))
        return (
          <span key={i} className="text-teal">
            {part}
          </span>
        );
      if (part.match(/^\d{4}-/))
        return (
          <span key={i} className="text-text-dim">
            {part}
          </span>
        );
      if (part.match(/^agent-/))
        return (
          <span key={i} className="text-accent">
            {part}
          </span>
        );
      return <span key={i}>{part}</span>;
    });
  }
  return text;
}

function RenderLine({ line }: { line: Line }) {
  const g = line.gap ? "mt-3" : "";

  if (line.type === "prompt") return null;

  if (line.type === "claude") {
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
      <span className={`block ${g} text-[#d8c1af]`}>
        {"  ⎿  "}
        {colorize(line.text)}
      </span>
    );
  }

  if (line.type === "output") {
    return (
      <span className={`block ${g} text-[#d8c1af]`}>
        {"     "}
        {colorize(line.text)}
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

export function SearchExamples() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  const [visibleCount, setVisibleCount] = useState(0);
  const timeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    if (!isInView) return;
    const prefersReducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) {
      setVisibleCount(lines.length);
      return;
    }

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
    <section id="examples" className="py-24">
      <div className="mx-auto max-w-[1120px] px-6">
        <div className="mb-3 text-center text-[11px] font-medium uppercase tracking-[0.12em] text-accent">
          Real Output
        </div>
        <h2 className="mx-auto mb-4 max-w-[680px] text-center font-display text-[clamp(26px,3.8vw,40px)] font-semibold leading-tight tracking-tight text-balance">
          Compact, formatted, zero noise
        </h2>
        <p className="mx-auto mb-14 max-w-[720px] text-center text-[15px] leading-relaxed font-light text-text-secondary">
          When a memory has no tags, the formatter omits the line. Output stays
          clean — same box-drawing format in Claude Code and BrainBar.
        </p>

        <motion.div
          ref={ref}
          aria-label="Terminal showing compact BrainLayer search"
          className="relative mx-auto max-w-[820px] overflow-hidden rounded-2xl border border-white/[0.06] bg-[#0c0c0e] shadow-[0_0_80px_rgba(212,149,106,0.04)]"
          initial={{ opacity: 1, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-40px" }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          {/* Title bar */}
          <div className="flex items-center gap-[7px] border-b border-white/[0.05] bg-white/[0.03] px-[18px] py-3.5">
            <div
              className="h-[11px] w-[11px] rounded-full bg-[#ff5f57]"
              aria-hidden="true"
            />
            <div
              className="h-[11px] w-[11px] rounded-full bg-[#febc2e]"
              aria-hidden="true"
            />
            <div
              className="h-[11px] w-[11px] rounded-full bg-[#28c840]"
              aria-hidden="true"
            />
            <span className="ml-2.5 font-mono text-xs text-text-dim">
              claude ~ another-project
            </span>
          </div>

          {/* Body — fixed height with internal scroll */}
          <div
            className="px-[22px] pt-5 pb-4 font-mono text-[13px] leading-[1.85] h-[340px] overflow-y-auto whitespace-pre-wrap"
            style={{
              scrollbarWidth: "thin",
              scrollbarColor: "#333338 transparent",
            }}
          >
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

          {/* Status bar */}
          <div className="px-[18px] font-mono text-text-dim">
            <div className="h-px bg-[#333338] mb-1" />
            <div className="flex items-center text-[#6ec1e4] py-0.5">
              <span>{"❯"}</span>
              <span className="ml-1 w-[7px] h-[13px] bg-text/30 animate-pulse" />
            </div>
            <div className="h-px bg-[#333338] mt-1 mb-1" />
            <div className="flex items-center justify-between pb-0.5 text-[10px]">
              <span>{"  ⎇ main | 🔧 3"}</span>
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
