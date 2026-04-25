"use client";

import { useState } from "react";

const ROUTER_URL = process.env.NEXT_PUBLIC_ROUTER_URL ?? "http://localhost:8000";

interface GenerateResponse {
  text: string;
  prompt_tokens: number;
  generated_tokens: number;
  prefill_worker: string;
  decode_worker: string;
}

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(128);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${ROUTER_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_new_tokens: maxTokens }),
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Router error ${res.status}: ${body}`);
      }

      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Generate</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-xs text-zinc-400 mb-1">Prompt</label>
          <textarea
            className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm font-mono resize-none focus:outline-none focus:ring-1 focus:ring-indigo-500"
            rows={5}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Once upon a time…"
          />
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs text-zinc-400">Max tokens</label>
          <input
            type="number"
            min={1}
            max={2048}
            className="w-24 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
          />
        </div>

        <button
          type="submit"
          disabled={loading || !prompt.trim()}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 rounded-lg text-sm font-medium transition-colors"
        >
          {loading ? "Generating…" : "Generate"}
        </button>
      </form>

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-3">
          <div className="bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3 font-mono text-sm whitespace-pre-wrap">
            {result.text}
          </div>
          <div className="flex gap-4 text-xs text-zinc-500">
            <span>prompt tokens: {result.prompt_tokens}</span>
            <span>generated: {result.generated_tokens}</span>
            <span>prefill: {result.prefill_worker}</span>
            <span>decode: {result.decode_worker}</span>
          </div>
        </div>
      )}
    </div>
  );
}
