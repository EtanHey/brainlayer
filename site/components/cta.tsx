"use client";

import { motion } from "framer-motion";
import { CopyBlock } from "./copy-block";

export function Cta() {
  return (
    <section className="relative py-24 text-center">
      <div
        className="pointer-events-none absolute bottom-[30%] left-1/2 h-[300px] w-[600px] -translate-x-1/2"
        style={{
          background:
            "radial-gradient(ellipse, rgba(212,149,106,0.06), transparent 70%)",
        }}
      />
      <div className="mx-auto max-w-[960px] px-6">
        <motion.h2
          className="mb-3 font-display text-[clamp(26px,4vw,42px)] font-semibold tracking-tight text-balance"
          initial={{ opacity: 1, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          Stop repeating yourself.
        </motion.h2>
        <motion.p
          className="mb-9 text-[15px] font-light text-text-secondary"
          initial={{ opacity: 1, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          Three commands. Persistent memory. Your data stays yours.
        </motion.p>
        <motion.div
          initial={{ opacity: 1, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <CopyBlock text="pip install brainlayer" showDollar />
        </motion.div>
      </div>
    </section>
  );
}
