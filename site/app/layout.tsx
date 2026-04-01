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
  title: "BrainLayer - Persistent Memory for MCP Agents",
  description:
    "Local-first memory for MCP agents and BrainBar. Twelve working tools, formatted Unicode output, and keyboard-first quick capture.",
  metadataBase: new URL("https://brainlayer.etanheyman.com"),
  alternates: { canonical: "/" },
  icons: {
    icon: "/favicon.ico",
    apple: "/apple-icon.png",
  },
  openGraph: {
    title: "BrainLayer - Persistent Memory for MCP Agents",
    description:
      "Local-first memory for MCP agents and BrainBar with twelve working tools and formatted search output.",
    url: "https://brainlayer.etanheyman.com",
    siteName: "BrainLayer",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "BrainLayer - Persistent Memory for MCP Agents",
    description:
      "Local-first memory for MCP agents and BrainBar. Twelve working MCP tools, formatted output, and F4 quick capture.",
  },
  robots: {
    index: true,
    follow: true,
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
