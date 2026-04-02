export function Footer() {
  const linkCls =
    "text-[13px] text-text-dim transition-colors hover:text-text-secondary focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none rounded";

  return (
    <footer className="border-t border-border py-10">
      <div className="mx-auto max-w-[960px] px-6">
        {/* Ecosystem cross-sell */}
        <div className="mb-8 flex flex-col items-center gap-4 sm:flex-row sm:justify-center sm:gap-8">
          <a
            href="https://voicelayer.etanheyman.com"
            className="group text-center sm:text-left"
          >
            <span className="text-[13px] font-medium text-text-secondary transition-colors group-hover:text-accent">
              VoiceLayer
            </span>
            <span className="ml-1.5 text-[12px] text-text-dim">
              &mdash; Voice I/O for AI agents
            </span>
          </a>
          <a
            href="https://cmuxlayer.etanheyman.com"
            className="group text-center sm:text-left"
          >
            <span className="text-[13px] font-medium text-text-secondary transition-colors group-hover:text-accent">
              cmuxLayer
            </span>
            <span className="ml-1.5 text-[12px] text-text-dim">
              &mdash; Terminal orchestration for AI agents
            </span>
          </a>
        </div>
        <p className="mb-6 text-center text-[12px] font-light text-text-dim">
          Pair with VoiceLayer for spoken knowledge capture.
        </p>

        {/* Standard links */}
        <div className="flex flex-col items-center justify-between gap-3 sm:flex-row">
          <div className="text-[13px] font-light text-text-dim">
            Built by{" "}
            <a
              href="https://etanheyman.com"
              className="text-text-secondary transition-colors hover:text-accent focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none rounded"
            >
              Etan Heyman
            </a>
            <span className="ml-2 text-border">|</span>
            <a
              href="https://etanheyman.com"
              className="ml-2 text-text-dim transition-colors hover:text-text-secondary"
            >
              Part of Golems
            </a>
          </div>
          <div className="flex flex-wrap justify-center gap-5">
            <a href="https://github.com/EtanHey/brainlayer" className={linkCls}>
              GitHub
            </a>
            <a href="https://etanhey.github.io/brainlayer" className={linkCls}>
              Docs
            </a>
            <a href="https://pypi.org/project/brainlayer/" className={linkCls}>
              PyPI
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
