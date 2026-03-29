import type { Metadata } from "next";
import { Newsreader, Outfit, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const newsreader = Newsreader({
  variable: "--font-display",
  subsets: ["latin"],
  style: ["normal", "italic"],
  weight: ["400", "600", "700"],
});

const outfit = Outfit({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "BrainLayer - Persistent Memory for AI Agents",
  description:
    "Local-first memory layer for MCP-compatible AI agents. 284K chunks indexed. Hybrid search in <50ms. Privacy-first.",
  openGraph: {
    title: "BrainLayer - Persistent Memory for AI Agents",
    description:
      "Local-first memory layer for MCP-compatible AI agents. Hybrid search in <50ms.",
    url: "https://brainlayer.etanheyman.com",
    siteName: "BrainLayer",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "BrainLayer - Persistent Memory for AI Agents",
    description:
      "Local-first memory for MCP agents. Every decision, correction, and preference - indexed and searchable in <50ms.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${newsreader.variable} ${outfit.variable} ${jetbrainsMono.variable} antialiased`}
    >
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
