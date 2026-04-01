"use client";

import { useState, useCallback } from "react";

interface CopyBlockProps {
  text: string;
  showDollar?: boolean;
  fullWidth?: boolean;
}

export function CopyBlock({ text, showDollar, fullWidth }: CopyBlockProps) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [text]);

  return (
    <div className={fullWidth ? "" : "mx-auto max-w-[420px]"}>
      <button
        onClick={copy}
        aria-label={copied ? "Copied" : "Copy to clipboard"}
        className={`group flex w-full items-center gap-3 rounded-[10px] border border-border bg-bg-card font-mono text-text-secondary transition-colors hover:border-accent ${fullWidth ? "px-3 py-2.5 text-xs" : "px-4 py-3 text-sm"}`}
      >
        <code className="min-w-0 flex-1 text-left text-text overflow-hidden text-ellipsis whitespace-nowrap">
          {showDollar && <span className="text-text-dim">$ </span>}
          {text}
        </code>
        <span className="flex h-5 w-5 shrink-0 items-center justify-center text-text-dim transition-colors group-hover:text-accent">
          {copied ? (
            <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              aria-hidden="true"
            >
              <path d="M3 8.5l3 3 7-7" />
            </svg>
          ) : (
            <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              aria-hidden="true"
            >
              <rect x="5" y="5" width="9" height="9" rx="1.5" />
              <path d="M5 11H3.5A1.5 1.5 0 012 9.5v-6A1.5 1.5 0 013.5 2h6A1.5 1.5 0 0111 3.5V5" />
            </svg>
          )}
        </span>
      </button>
    </div>
  );
}
