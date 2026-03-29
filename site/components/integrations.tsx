"use client";

import { motion } from "framer-motion";
import Image from "next/image";
import { CopyBlock } from "./copy-block";

const logos = [
  { name: "Claude Code", src: "/logos/claude.svg" },
  { name: "Cursor", src: "/logos/cursor.svg" },
  { name: "Zed", src: "/logos/zed.svg" },
  { name: "VS Code", src: "/logos/vscode.svg" },
  { name: "Codex", src: "/logos/openai.svg" },
  { name: "Kiro", src: "/logos/kiro.svg" },
  { name: "Gemini CLI", src: "/logos/gemini.svg" },
];

const steps = [
  { num: "01", text: "Install from PyPI", cmd: "pip install brainlayer" },
  {
    num: "02",
    text: "Configure MCP and index conversations",
    cmd: "brainlayer init",
  },
  { num: "03", text: "Start real-time indexing", cmd: "brainlayer watch" },
];

export function Integrations() {
  return (
    <section id="setup" className="py-20 text-center">
      <div className="mx-auto max-w-[960px] px-6">
        <div className="mb-3 text-[11px] font-medium uppercase tracking-[0.12em] text-accent">
          Works with
        </div>
        <h2 className="mb-12 font-display text-[clamp(26px,3.5vw,36px)] font-semibold leading-tight tracking-tight">
          Any MCP client
        </h2>

        <motion.div
          className="mb-16 flex flex-wrap justify-center gap-10"
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          {logos.map((logo) => (
            <div
              key={logo.name}
              className="group flex flex-col items-center gap-2.5 transition-transform hover:-translate-y-[3px]"
            >
              <div className="flex h-[52px] w-[52px] items-center justify-center rounded-xl border border-border bg-bg-card p-[11px] transition-colors group-hover:border-border-hover">
                <Image
                  src={logo.src}
                  alt={logo.name}
                  width={30}
                  height={30}
                  className="h-full w-full object-contain"
                />
              </div>
              <span className="text-xs text-text-dim transition-colors group-hover:text-text-secondary">
                {logo.name}
              </span>
            </div>
          ))}
        </motion.div>

        <div className="mb-3 text-[11px] font-medium uppercase tracking-[0.12em] text-accent">
          Getting started
        </div>
        <h2 className="mb-12 font-display text-[clamp(26px,3.5vw,36px)] font-semibold leading-tight tracking-tight">
          Three steps
        </h2>

        <motion.div
          className="mx-auto grid max-w-[780px] grid-cols-1 gap-6 md:grid-cols-3"
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          {steps.map((step) => (
            <div key={step.num} className="flex flex-col text-left">
              <div className="mb-2.5 font-mono text-[13px] font-medium text-accent">
                {step.num}
              </div>
              <p className="mb-2.5 flex-1 text-sm font-light text-text-secondary">
                {step.text}
              </p>
              <CopyBlock text={step.cmd} fullWidth />
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
