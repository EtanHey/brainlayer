export function Footer() {
  return (
    <footer className="border-t border-border py-8">
      <div className="mx-auto flex max-w-[960px] flex-col items-center justify-between gap-3 px-6 sm:flex-row">
        <div className="text-[13px] font-light text-text-dim">
          Built by{" "}
          <a
            href="https://etanheyman.com"
            className="text-text-secondary transition-colors hover:text-accent focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none rounded"
          >
            Etan Heyman
          </a>
        </div>
        <div className="flex gap-5">
          <a
            href="https://github.com/EtanHey/brainlayer"
            className="text-[13px] text-text-dim transition-colors hover:text-text-secondary focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none rounded"
          >
            GitHub
          </a>
          <a
            href="https://etanhey.github.io/brainlayer"
            className="text-[13px] text-text-dim transition-colors hover:text-text-secondary focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none rounded"
          >
            Docs
          </a>
          <a
            href="https://pypi.org/project/brainlayer/"
            className="text-[13px] text-text-dim transition-colors hover:text-text-secondary focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none rounded"
          >
            PyPI
          </a>
        </div>
      </div>
    </footer>
  );
}
