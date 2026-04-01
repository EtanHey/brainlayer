import { CopyBlock } from "./copy-block";

export function Hero() {
  return (
    <section className="relative pt-44 pb-20 text-center">
      {/* Ambient glow - wide and soft, no hard edges */}
      <div
        className="pointer-events-none absolute top-0 left-1/2 h-[700px] w-[1400px] -translate-x-1/2"
        style={{
          background:
            "radial-gradient(ellipse at 45% 35%, rgba(212,149,106,0.045), transparent 60%), radial-gradient(ellipse at 55% 65%, rgba(94,234,212,0.025), transparent 60%)",
          filter: "blur(40px)",
        }}
      />

      <div className="relative mx-auto max-w-[960px] px-6">
        <h1 className="hero-fade mx-auto mb-6 max-w-[700px] font-display text-[clamp(40px,6vw,68px)] leading-[1.08] font-bold tracking-[-0.035em] text-balance">
          BrainLayer remembers
          <br />
          the work you already did.
        </h1>

        <p className="hero-fade hero-fade-d1 mx-auto mb-10 max-w-[480px] text-[17px] leading-relaxed font-light text-text-secondary">
          Local-first memory for MCP agents and BrainBar. Twelve working tools,
          formatted Unicode output, keyboard-first capture, and one SQLite file
          that stays on your machine.
        </p>

        <div className="hero-fade hero-fade-d2 mb-12 flex flex-wrap items-center justify-center gap-3">
          <a
            href="#setup"
            className="inline-flex items-center gap-2 rounded-full bg-text px-6 py-3 text-sm font-medium text-bg transition-transform hover:scale-[1.03] hover:shadow-[0_0_24px_rgba(250,250,249,0.15)] active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none"
          >
            Get started
          </a>
          <a
            href="https://github.com/EtanHey/brainlayer"
            className="inline-flex items-center gap-2 rounded-full border border-border px-6 py-3 text-sm font-medium text-text-secondary transition-all hover:scale-[1.03] hover:border-border-hover hover:text-text active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            View source
          </a>
        </div>

        <div className="hero-fade hero-fade-d3">
          <CopyBlock text="pip install brainlayer" showDollar />
        </div>

        <div className="hero-fade hero-fade-d4 mt-8 flex flex-wrap items-center justify-center gap-2 text-[12px] text-text-dim">
          <span className="rounded-full border border-border px-3 py-1">
            12 working MCP tools
          </span>
          <span className="rounded-full border border-border px-3 py-1">
            F4 quick capture
          </span>
          <span className="rounded-full border border-border px-3 py-1">
            dashboard popover
          </span>
          <span className="rounded-full border border-border px-3 py-1">
            no tag bloat
          </span>
        </div>
      </div>
    </section>
  );
}
