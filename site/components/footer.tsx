const ECOSYSTEM = [
  {
    name: "BrainLayer",
    desc: "Persistent memory for AI agents",
    href: "https://brainlayer.etanheyman.com",
    current: true,
  },
  {
    name: "VoiceLayer",
    desc: "Voice I/O for AI agents",
    href: "https://voicelayer.etanheyman.com",
    current: false,
  },
  {
    name: "cmuxLayer",
    desc: "Terminal orchestration for AI agents",
    href: "https://cmuxlayer.etanheyman.com",
    current: false,
  },
];

const LINKS = [
  { label: "GitHub", href: "https://github.com/EtanHey/brainlayer" },
  { label: "Docs", href: "https://etanhey.github.io/brainlayer" },
  { label: "PyPI", href: "https://pypi.org/project/brainlayer/" },
];

export function Footer() {
  return (
    <footer className="border-t border-border py-10">
      <div className="mx-auto max-w-[960px] px-6">
        {/* Ecosystem section */}
        <div className="mb-8">
          <div className="mb-4 font-mono text-[10px] uppercase tracking-[0.12em] text-text-dim">
            Golems Ecosystem
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {ECOSYSTEM.map((p) => (
              <a
                key={p.name}
                href={p.current ? "#" : p.href}
                className={`text-[13px] no-underline transition-colors ${
                  p.current
                    ? "cursor-default font-medium text-text"
                    : "text-text-dim hover:text-text-secondary"
                }`}
              >
                {p.name}
                <span className="mt-0.5 block text-[11px] font-light text-text-dim">
                  {p.desc}
                </span>
              </a>
            ))}
          </div>
          <p className="mt-4 text-[11px] font-light text-text-dim">
            Three open-source MCP servers. One agent toolkit.
          </p>
        </div>

        {/* Bottom row */}
        <div className="flex items-center justify-between border-t border-border pt-4 max-md:flex-col max-md:gap-3">
          <div className="text-[13px] font-light text-text-dim">
            Built by{" "}
            <a
              href="https://etanheyman.com"
              className="text-text-secondary no-underline transition-colors hover:text-accent"
            >
              Etan Heyman
            </a>
          </div>
          <div className="flex gap-5">
            {LINKS.map((link) => (
              <a
                key={link.label}
                href={link.href}
                target="_blank"
                rel="noopener"
                className="text-[13px] text-text-dim no-underline transition-colors hover:text-text-secondary"
              >
                {link.label}
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
