import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Disagg Serving",
  description: "Playground for disaggregated prefill/decode inference",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-zinc-950 text-zinc-100 min-h-screen antialiased">
        <header className="border-b border-zinc-800 px-6 py-3 flex items-center gap-3">
          <span className="text-sm font-mono font-semibold text-indigo-400">llm-disagg</span>
          <span className="text-zinc-600 text-xs">prefill / decode playground</span>
        </header>
        <main className="max-w-3xl mx-auto px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
