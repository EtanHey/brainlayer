"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";

export function Nav() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 py-4 backdrop-blur-xl transition-colors duration-300 ${
        scrolled
          ? "border-b border-border bg-bg/80"
          : "border-b border-transparent bg-bg/80"
      }`}
    >
      <div className="mx-auto flex max-w-[960px] items-center justify-between px-6">
        <Link
          href="/"
          className="flex items-center gap-2.5 text-text opacity-90 transition-opacity hover:opacity-100 focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none rounded"
        >
          <Image
            src="/logos/brainlayer.svg"
            alt="BrainLayer home"
            width={24}
            height={24}
            className="hue-rotate-[120deg] saturate-[0.65] brightness-110"
          />
          <span className="font-sans text-base font-semibold tracking-tight">
            BrainLayer
          </span>
        </Link>
        <div className="flex items-center gap-6">
          <a
            href="#tools"
            className="hidden text-sm text-text-secondary transition-colors hover:text-text sm:inline focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none rounded"
          >
            Tools
          </a>
          <a
            href="#setup"
            className="hidden text-sm text-text-secondary transition-colors hover:text-text sm:inline focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none rounded"
          >
            Setup
          </a>
          <Link
            href="/docs"
            className="hidden text-sm text-text-secondary transition-colors hover:text-text sm:inline focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none rounded"
          >
            Docs
          </Link>
          <a
            href="https://github.com/EtanHey/brainlayer"
            className="flex items-center gap-1.5 text-sm text-text-secondary transition-colors hover:text-text focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none rounded"
          >
            GitHub
            <svg
              width="12"
              height="12"
              viewBox="0 0 12 12"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              className="ml-0.5 opacity-50"
              aria-hidden="true"
            >
              <path d="M3 9L9 3M9 3H4.5M9 3v4.5" />
            </svg>
          </a>
        </div>
      </div>
    </nav>
  );
}
