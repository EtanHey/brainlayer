"use client";

import { useEffect, useState } from "react";
import Image from "next/image";

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
        <a
          href="#"
          className="flex items-center gap-2.5 text-text opacity-90 transition-opacity hover:opacity-100"
        >
          <Image
            src="/logos/brainlayer.svg"
            alt=""
            width={24}
            height={24}
            className="hue-rotate-[120deg] saturate-[0.65] brightness-110"
          />
          <span className="font-sans text-base font-semibold tracking-tight">
            BrainLayer
          </span>
        </a>
        <div className="flex items-center gap-6">
          <a
            href="#tools"
            className="hidden text-sm text-text-secondary transition-colors hover:text-text sm:inline"
          >
            Tools
          </a>
          <a
            href="#setup"
            className="hidden text-sm text-text-secondary transition-colors hover:text-text sm:inline"
          >
            Setup
          </a>
          <a
            href="/docs"
            className="hidden text-sm text-text-secondary transition-colors hover:text-text sm:inline"
          >
            Docs
          </a>
          <a
            href="https://github.com/EtanHey/brainlayer"
            className="flex items-center gap-1.5 text-sm text-text-secondary transition-colors hover:text-text"
          >
            GitHub{" "}
            <span className="inline-block transition-transform group-hover:translate-x-0.5">
              ↗
            </span>
          </a>
        </div>
      </div>
    </nav>
  );
}
