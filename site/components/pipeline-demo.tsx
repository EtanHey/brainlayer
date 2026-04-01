"use client";

import { useRef, useEffect, useCallback } from "react";
import { motion, useInView } from "framer-motion";

/* ── Graph data ─────────────────────────────────────────────── */

interface Node {
  id: string;
  x: number;
  y: number;
  r: number;
  label: string;
  color: string;
}

const nodes: Node[] = [
  { id: "auth", x: 160, y: 60, r: 8, label: "auth", color: "#8B5CF6" },
  {
    id: "extract",
    x: 90,
    y: 130,
    r: 10,
    label: "extraction",
    color: "#8B5CF6",
  },
  { id: "6pm", x: 240, y: 110, r: 7, label: "6pm-mini", color: "#d4956a" },
  { id: "twin", x: 140, y: 200, r: 9, label: "twin-primary", color: "#8B5CF6" },
  { id: "decision", x: 60, y: 60, r: 6, label: "decision", color: "#d4956a" },
  { id: "convex", x: 250, y: 200, r: 7, label: "convex", color: "#5eead4" },
  { id: "session", x: 50, y: 190, r: 5, label: "session", color: "#5eead4" },
  { id: "broker", x: 200, y: 250, r: 6, label: "broker", color: "#8B5CF6" },
  { id: "slot", x: 100, y: 260, r: 5, label: "slot-confirm", color: "#d4956a" },
];

const edges: [string, string][] = [
  ["extract", "twin"],
  ["extract", "auth"],
  ["twin", "6pm"],
  ["decision", "extract"],
  ["6pm", "convex"],
  ["auth", "6pm"],
  ["twin", "broker"],
  ["session", "extract"],
  ["broker", "slot"],
  ["convex", "twin"],
  ["decision", "auth"],
];

/* ── Helpers ────────────────────────────────────────────────── */

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// Color span builders
const tool = (s: string) => `<span class="dt-tool">${s}</span>`;
const str = (s: string) => `<span class="dt-str">${s}</span>`;
const num = (s: string) => `<span class="dt-num">${s}</span>`;
const dim = (s: string) => `<span class="dt-dim">${s}</span>`;
const key = (s: string) => `<span class="dt-key">${s}</span>`;
const bdr = (s: string) => `<span class="dt-border">${s}</span>`;
const body = (s: string) => `<span class="dt-body">${s}</span>`;
const tag = (s: string) => `<span class="dt-tag">${s}</span>`;
const score = (s: string) => `<span class="dt-score">${s}</span>`;

/* ── Component ──────────────────────────────────────────────── */

export function PipelineDemo() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const layoutRef = useRef<HTMLDivElement>(null);
  const sessionLabelRef = useRef<HTMLSpanElement>(null);
  const sessionIdRef = useRef<HTMLSpanElement>(null);
  const entityCountRef = useRef<HTMLSpanElement>(null);
  const dsChunksRef = useRef<HTMLSpanElement>(null);
  const dsEntitiesRef = useRef<HTMLSpanElement>(null);
  const dsLatencyRef = useRef<HTMLSpanElement>(null);
  const dsStatusRef = useRef<HTMLSpanElement>(null);
  const cancelledRef = useRef(false);

  const isInView = useInView(sectionRef, { once: true, margin: "-80px" });

  /* ── Imperative animation engine ───────────────────────── */

  const runAnimation = useCallback(async () => {
    const t = terminalRef.current;
    const s = svgRef.current;
    const l = layoutRef.current;
    if (!t || !s || !l) return;
    // Non-null bindings for use in nested closures
    const terminal: HTMLDivElement = t;
    const svg: SVGSVGElement = s;
    const layout: HTMLDivElement = l;

    const cancelled = () => cancelledRef.current;
    const guardSleep = async (ms: number) => {
      await sleep(ms);
      if (cancelled()) throw new Error("cancelled");
    };

    function append(html: string) {
      const span = document.createElement("span");
      span.className = "dt-line";
      span.innerHTML = html;
      terminal.appendChild(span);
    }

    function gap() {
      const span = document.createElement("span");
      span.className = "dt-line dt-gap";
      terminal.appendChild(span);
    }

    async function typeIn(text: string, speed = 30) {
      const span = document.createElement("span");
      span.className = "dt-line dt-user";
      terminal.appendChild(span);
      const cursor = '<span class="dt-cursor"></span>';
      for (let i = 0; i <= text.length; i++) {
        if (cancelled()) throw new Error("cancelled");
        span.innerHTML = esc(text.slice(0, i)) + cursor;
        await sleep(speed + Math.random() * speed * 0.5);
      }
      await sleep(200);
      span.innerHTML = esc(text);
    }

    function resetGraph() {
      svg.innerHTML = "";
      edges.forEach(([from, to], i) => {
        const a = nodes.find((n) => n.id === from)!;
        const b = nodes.find((n) => n.id === to)!;
        const line = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "line",
        );
        line.setAttribute("x1", String(a.x));
        line.setAttribute("y1", String(a.y));
        line.setAttribute("x2", String(b.x));
        line.setAttribute("y2", String(b.y));
        line.setAttribute("stroke", "#8B5CF6");
        line.setAttribute("stroke-width", "1");
        line.setAttribute("stroke-opacity", "0.12");
        line.classList.add("kg-edge");
        line.id = "edge-" + i;
        svg.appendChild(line);
      });
      nodes.forEach((n) => {
        const c = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "circle",
        );
        c.setAttribute("cx", String(n.x));
        c.setAttribute("cy", String(n.y));
        c.setAttribute("r", String(n.r));
        c.setAttribute("fill", n.color);
        c.setAttribute("opacity", "0");
        c.classList.add("kg-node");
        c.id = "node-" + n.id;
        svg.appendChild(c);

        const t = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "text",
        );
        t.setAttribute("x", String(n.x));
        t.setAttribute("y", String(n.y + n.r + 12));
        t.classList.add("kg-label");
        t.id = "label-" + n.id;
        t.textContent = n.label;
        svg.appendChild(t);
      });
    }

    function showNode(id: string, pulse?: boolean) {
      const node = document.getElementById("node-" + id);
      if (!node) return;
      node.style.opacity = "1";
      node.classList.add("visible");
      if (pulse) node.classList.add("kg-pulse");
      const label = document.getElementById("label-" + id);
      if (label) label.classList.add("visible");
    }

    function highlightNode(id: string) {
      const node = document.getElementById("node-" + id);
      if (!node) return;
      const orig = node.getAttribute("fill")!;
      node.setAttribute("fill", "#e8b090");
      node.setAttribute("r", String(parseFloat(node.getAttribute("r")!) + 3));
      setTimeout(() => {
        node.setAttribute("fill", orig);
        node.setAttribute("r", String(parseFloat(node.getAttribute("r")!) - 3));
      }, 2000);
    }

    function drawEdge(idx: number) {
      const e = document.getElementById("edge-" + idx);
      if (e) {
        e.setAttribute("stroke-opacity", "0.4");
        e.classList.add("drawn");
      }
    }

    let baseChunks = 284291;
    let baseEntities = 12847;

    function updateStats(
      chunks: number | null,
      entities: number | null,
      latency: string | null,
      status: string | null,
    ) {
      if (chunks !== null && dsChunksRef.current)
        dsChunksRef.current.textContent = chunks.toLocaleString();
      if (entities !== null && dsEntitiesRef.current)
        dsEntitiesRef.current.textContent = entities.toLocaleString();
      if (latency !== null && dsLatencyRef.current)
        dsLatencyRef.current.textContent = latency;
      if (status !== null && dsStatusRef.current)
        dsStatusRef.current.textContent = status;
    }

    function setEntityCount(n: string | number) {
      if (entityCountRef.current)
        entityCountRef.current.textContent = n + " entities";
    }

    function setSession(label: string, id?: string) {
      if (sessionLabelRef.current) sessionLabelRef.current.textContent = label;
      if (id && sessionIdRef.current) sessionIdRef.current.textContent = id;
    }

    /* ── Run one full cycle ────────────────────────────── */

    async function runCycle() {
      terminal.innerHTML = "";
      resetGraph();
      baseChunks++;
      baseEntities += 3;

      // Phase 1: Store a decision
      setSession("Session 1 \u2014 storing", "sess-a7f3c");
      updateStats(null, null, null, "indexing");

      append(dim("$") + " " + body("Claude Code \u2014 6pm-mini project"));
      gap();
      await guardSleep(500);

      await typeIn("Store this architecture decision.", 30);
      gap();
      await guardSleep(300);

      append(
        bdr("\u250C\u2500") +
          " " +
          tool("brain_store") +
          "(" +
          key("content") +
          "=..., " +
          key("tags") +
          "=..., " +
          key("importance") +
          "=" +
          num("8") +
          ")",
      );
      append(bdr("\u2502"));
      await guardSleep(300);
      append(bdr("\u2502") + "  " + str('"Chose twin-primary extraction over'));
      append(
        bdr("\u2502") + "   " + str("broker-primary \u2014 twins extract in"),
      );
      append(bdr("\u2502") + "   " + str('real-time during conversation"'));
      append(bdr("\u2502"));
      await guardSleep(400);

      showNode("extract", true);
      showNode("twin", true);
      setEntityCount(2);
      await guardSleep(500);

      showNode("decision", true);
      drawEdge(3);
      setEntityCount(3);
      await guardSleep(400);

      showNode("6pm", false);
      drawEdge(0);
      drawEdge(2);
      setEntityCount(4);

      append(
        bdr("\u2502") + "  " + dim("stored:") + " " + str('"rt-a4f82c19"'),
      );
      append(
        bdr("\u2502") +
          "  " +
          dim("tags:") +
          " " +
          tag("decision") +
          " " +
          tag("6pm"),
      );
      append(
        bdr("\u2502") +
          "  " +
          dim("importance:") +
          " " +
          num("8") +
          "  " +
          dim("entities:") +
          " " +
          num("3"),
      );
      append(bdr("\u2514\u2500") + " " + dim("12ms"));
      gap();
      updateStats(baseChunks, baseEntities, "12ms", "stored");
      await guardSleep(1500);

      // Phase 2: Graph grows
      showNode("auth", false);
      drawEdge(1);
      drawEdge(5);
      setEntityCount(5);
      await guardSleep(600);

      showNode("convex", false);
      drawEdge(4);
      drawEdge(9);
      setEntityCount(6);
      await guardSleep(800);

      showNode("broker", false);
      drawEdge(6);
      setEntityCount(7);
      await guardSleep(600);

      // Phase 3: New session searches
      terminal.innerHTML = "";
      setSession("Session 2 \u2014 searching", "sess-e92b1");
      updateStats(null, null, null, "searching");

      append(dim("$") + " " + body("New session \u2014 different project"));
      gap();
      await guardSleep(500);

      await typeIn("What approach did we use for extraction?", 28);
      gap();
      await guardSleep(300);

      append(
        bdr("\u250C\u2500") +
          " " +
          tool("brain_search") +
          "(" +
          key("query") +
          "=" +
          str('"extraction architecture"') +
          ")",
      );
      append(bdr("\u2502"));
      await guardSleep(300);

      updateStats(null, null, "9ms", null);
      append(
        bdr("\u2502") +
          "  " +
          dim("chunks:") +
          " " +
          num("3") +
          "  " +
          dim("latency:") +
          " " +
          num("9ms") +
          "  " +
          dim("top:") +
          " " +
          score("0.94"),
      );
      append(bdr("\u2502"));
      await guardSleep(500);

      highlightNode("extract");
      highlightNode("twin");

      append(
        bdr("\u2502") +
          "  " +
          str('"Chose twin-primary extraction over broker-primary.'),
      );
      append(
        bdr("\u2502") +
          "   " +
          str("Reason: twins extract in real-time during conversation."),
      );
      append(
        bdr("\u2502") + "   " + str('Decision: 2026-03-14. Confidence: high."'),
      );
      append(bdr("\u2502"));
      append(
        bdr("\u2502") +
          "  " +
          dim("tags:") +
          " " +
          tag("decision") +
          " " +
          tag("6pm"),
      );
      append(
        bdr("\u2514\u2500") +
          " " +
          dim("rt-a4f82c19 \u00B7 importance:") +
          " " +
          num("8"),
      );
      gap();
      await guardSleep(2000);

      // Phase 4: brain_recall
      terminal.innerHTML = "";
      setSession("Session 2 \u2014 recalling");
      updateStats(null, null, null, "context");

      await typeIn("Show me the full session context.", 28);
      gap();
      await guardSleep(300);

      append(
        bdr("\u250C\u2500") +
          " " +
          tool("brain_recall") +
          "(" +
          key("mode") +
          "=" +
          str('"context"') +
          ")",
      );
      append(bdr("\u2502"));
      await guardSleep(400);

      append(
        bdr("\u2502") + "  " + dim("session:") + " " + str('"sess-e92b1"'),
      );
      append(bdr("\u2502") + "  " + dim("chunks injected:") + " " + num("4"));
      append(bdr("\u2502") + "  " + dim("tokens used:") + " " + num("1,247"));
      append(bdr("\u2502") + "  " + dim("decisions found:") + " " + num("2"));
      append(bdr("\u2502") + "  " + dim("corrections:") + " " + num("1"));
      append(
        bdr("\u2514\u2500") + " " + dim("context loaded in") + " " + num("8ms"),
      );
      gap();
      updateStats(null, null, "8ms", "recalled");

      showNode("session", false);
      drawEdge(7);
      showNode("slot", false);
      drawEdge(8);
      drawEdge(10);
      setEntityCount(9);
      await guardSleep(2000);

      // Phase 5: brain_digest
      terminal.innerHTML = "";
      setSession("Session 2 \u2014 digesting");
      updateStats(null, null, null, "digesting");

      await typeIn("Digest this architecture doc.", 28);
      gap();
      await guardSleep(300);

      append(
        bdr("\u250C\u2500") +
          " " +
          tool("brain_digest") +
          "(" +
          key("content") +
          "=" +
          str('"<2400 words>"') +
          ")",
      );
      append(bdr("\u2502"));
      await guardSleep(400);

      append(bdr("\u2502") + "  " + dim("extracting entities..."));
      await guardSleep(600);
      append(
        bdr("\u2502") +
          "  " +
          dim("\u2192 entities:") +
          " " +
          num("7") +
          "  " +
          dim("relations:") +
          " " +
          num("12"),
      );
      await guardSleep(400);
      append(
        bdr("\u2502") + "  " + dim("\u2192 chunks created:") + " " + num("4"),
      );
      append(
        bdr("\u2502") +
          "  " +
          dim("\u2192 tags auto-applied:") +
          " " +
          tag("architecture") +
          " " +
          tag("extraction") +
          " " +
          tag("convex"),
      );
      append(
        bdr("\u2514\u2500") + " " + dim("digested in") + " " + num("340ms"),
      );
      gap();

      updateStats(baseChunks + 4, baseEntities + 7, "340ms", "digested");
      setEntityCount("9 +7");
      await guardSleep(2000);

      // Phase 6: Stats settle
      updateStats(baseChunks + 4, baseEntities + 7, "11ms", "idle");
      await guardSleep(1500);
    }

    /* ── Animation loop ───────────────────────────────── */

    const reducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (reducedMotion) {
      try {
        await runCycle();
      } catch {
        /* cancelled */
      }
      return;
    }

    while (!cancelled()) {
      try {
        layout.style.opacity = "1";
        await runCycle();
        if (cancelled()) break;
        layout.style.opacity = "0.15";
        await guardSleep(1200);
      } catch {
        break;
      }
    }
  }, []);

  useEffect(() => {
    if (!isInView) return;
    cancelledRef.current = false;
    runAnimation();
    return () => {
      cancelledRef.current = true;
    };
  }, [isInView, runAnimation]);

  return (
    <section id="demo" className="py-24" ref={sectionRef}>
      <div className="mx-auto max-w-[1120px] px-6">
        <div className="mb-3 text-center text-[11px] font-medium uppercase tracking-[0.12em] text-accent">
          See it work
        </div>
        <h2 className="mx-auto mb-4 max-w-[680px] text-center font-display text-[clamp(26px,3.8vw,40px)] font-semibold leading-tight tracking-tight text-balance">
          The memory pipeline in action
        </h2>
        <p className="mx-auto mb-14 max-w-[720px] text-center text-[15px] leading-relaxed font-light text-text-secondary">
          Store decisions, search across sessions, recall context, digest
          documents. One pipeline, twelve tools, zero configuration.
        </p>

        <motion.div
          className="relative mx-auto max-w-[920px] overflow-hidden rounded-2xl border border-white/[0.06] bg-[#0c0c0e] shadow-[0_0_80px_rgba(212,149,106,0.04)]"
          initial={{ opacity: 0, y: 24 }}
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
              brainlayer &mdash; memory pipeline
            </span>
          </div>

          {/* Split pane layout */}
          <div
            ref={layoutRef}
            className="flex min-h-[360px] flex-col transition-opacity duration-600 md:flex-row"
          >
            {/* Left: Terminal pane */}
            <div className="flex flex-[0_0_58%] flex-col border-b border-white/[0.05] md:border-r md:border-b-0">
              <div className="flex items-center justify-between border-b border-white/[0.04] px-4 py-2 font-mono text-[11px] text-text-dim">
                <span
                  ref={sessionLabelRef}
                  className="font-medium text-text-secondary"
                >
                  Session 1 &mdash; storing
                </span>
                <span ref={sessionIdRef} className="text-text-dim">
                  sess-a7f3c
                </span>
              </div>
              <div
                ref={terminalRef}
                className="demo-terminal flex-1 overflow-hidden px-[18px] py-4 font-mono text-[12.5px] leading-[1.8]"
              />
            </div>

            {/* Right: Knowledge graph pane */}
            <div className="flex min-h-[240px] flex-1 flex-col items-center justify-center">
              <div className="flex w-full items-center justify-between border-b border-white/[0.04] px-4 py-2 font-mono text-[11px] text-text-dim">
                <span className="font-medium text-text-secondary">
                  Knowledge Graph
                </span>
                <span ref={entityCountRef}>0 entities</span>
              </div>
              <div className="relative flex-1 w-full p-5">
                <svg
                  ref={svgRef}
                  viewBox="0 0 320 280"
                  className="h-full w-full"
                />
              </div>
            </div>
          </div>

          {/* Stats bar */}
          <div className="flex flex-wrap gap-x-6 gap-y-1 border-t border-white/[0.05] bg-white/[0.02] px-[18px] py-2.5 font-mono text-[11px] text-text-dim">
            <span>
              <span className="mr-1">chunks</span>
              <span ref={dsChunksRef} className="font-medium text-teal">
                284,291
              </span>
            </span>
            <span>
              <span className="mr-1">entities</span>
              <span ref={dsEntitiesRef} className="font-medium text-teal">
                12,847
              </span>
            </span>
            <span>
              <span className="mr-1">latency</span>
              <span ref={dsLatencyRef} className="font-medium text-teal">
                11ms
              </span>
            </span>
            <span>
              <span className="mr-1">pipeline</span>
              <span ref={dsStatusRef} className="font-medium text-teal">
                idle
              </span>
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
