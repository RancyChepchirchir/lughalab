"use client";

import { useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  BrainCircuit,
  Loader2,
  Languages,
  Network,
  ScanText,
} from "lucide-react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type TokenRepresentation = {
  token: string;
  embedding_norm?: number;
};

type AnalyseResult = {
  text: string;
  model: string;
  language: string;
  device: string;
  tokens: TokenRepresentation[] | string[];
  token_count: number;
  embedding_dimension: number;
  sentence_embedding: number[];
  embedding_norm: number;
  inference_time_ms: number;
  note: string;
};

type POSToken = {
  token: string;
  tag: string;
  confidence: number;
};

type POSResult = {
  text: string;
  model: string;
  language: string;
  device?: string;
  tokens: POSToken[];
  inference_time_ms?: number;
  note?: string;
};

export default function NLPLabPage() {
  const [text, setText] = useState(
    "Elimu ni muhimu kwa maendeleo ya jamii."
  );

  const [analyseResult, setAnalyseResult] =
    useState<AnalyseResult | null>(null);

  const [posResult, setPosResult] =
    useState<POSResult | null>(null);

  const [loadingAnalyse, setLoadingAnalyse] = useState(false);
  const [loadingPOS, setLoadingPOS] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function analyseRepresentation() {
    setLoadingAnalyse(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/nlp/analyse`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(
          `Representation request failed (${response.status})`
        );
      }

      const data = await response.json();
      setAnalyseResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to analyse the sentence."
      );
    } finally {
      setLoadingAnalyse(false);
    }
  }

  async function analysePOS() {
    setLoadingPOS(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/nlp/pos`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`POS request failed (${response.status})`);
      }

      const data = await response.json();
      setPosResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to perform POS analysis."
      );
    } finally {
      setLoadingPOS(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#080a0d] text-[#f1efe8]">
      <header className="border-b border-white/10">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6 lg:px-10">
          <Link
            href="/"
            className="flex items-center gap-2 text-sm text-white/45 transition hover:text-white"
          >
            <ArrowLeft size={15} />
            LughaLab
          </Link>

          <div className="flex items-center gap-3">
            <Languages size={17} className="text-emerald-300/70" />
            <span className="font-mono text-xs uppercase tracking-[0.2em] text-white/45">
              LughaNLP
            </span>
          </div>
        </div>
      </header>

      <section className="border-b border-white/10">
        <div className="mx-auto max-w-7xl px-6 py-16 lg:px-10 lg:py-20">
          <p className="mb-4 font-mono text-xs uppercase tracking-[0.25em] text-emerald-300/65">
            Text / 01
          </p>

          <h1 className="max-w-4xl text-5xl tracking-[-0.045em] sm:text-6xl">
            Swahili
            <span className="ml-3 text-white/30">
              Representation Laboratory
            </span>
          </h1>

          <p className="mt-7 max-w-2xl text-sm leading-7 text-white/45">
            Inspect contextual representations and linguistic predictions
            from pretrained African-language models. Representation
            similarity is treated as a diagnostic rather than evidence of
            semantic understanding.
          </p>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-8 px-6 py-12 lg:grid-cols-[0.9fr_1.1fr] lg:px-10">
        <div>
          <div className="border border-white/10 p-6">
            <div className="mb-5 flex items-center gap-3">
              <ScanText size={17} className="text-white/45" />
              <h2 className="text-sm font-medium">
                Input sentence
              </h2>
            </div>

            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows={7}
              className="w-full resize-none border border-white/10 bg-white/[0.025] p-4 text-sm leading-7 text-white outline-none transition placeholder:text-white/20 focus:border-emerald-300/40"
            />

            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <button
                onClick={analyseRepresentation}
                disabled={loadingAnalyse || !text.trim()}
                className="flex items-center justify-center gap-2 bg-[#f1efe8] px-4 py-3 text-xs font-medium text-[#080a0d] transition hover:bg-white disabled:opacity-40"
              >
                {loadingAnalyse ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <BrainCircuit size={14} />
                )}
                Analyse representation
              </button>

              <button
                onClick={analysePOS}
                disabled={loadingPOS || !text.trim()}
                className="flex items-center justify-center gap-2 border border-white/15 px-4 py-3 text-xs text-white/65 transition hover:border-white/30 hover:text-white disabled:opacity-40"
              >
                {loadingPOS ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Network size={14} />
                )}
                Run POS analysis
              </button>
            </div>

            {error && (
              <div className="mt-5 border border-red-400/20 bg-red-400/[0.05] p-4 text-xs leading-5 text-red-200/75">
                {error}
              </div>
            )}
          </div>

          <div className="mt-6 border border-white/10 p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/30">
              Research note
            </p>

            <div className="mt-5 space-y-4 text-sm leading-6 text-white/40">
              <p>
                Contextual token states are mean-pooled using the attention
                mask to obtain a sentence-level representation.
              </p>

              <div className="overflow-x-auto border-l border-emerald-300/30 py-2 pl-4 font-mono text-xs text-white/60">
                z = Σ mᵢhᵢ / Σ mᵢ
              </div>

              <p>
                A 768-dimensional vector is therefore a representation of
                the sentence under the encoder, not a direct measurement of
                meaning.
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          {!analyseResult && !posResult && (
            <div className="flex min-h-[420px] items-center justify-center border border-dashed border-white/10">
              <div className="max-w-sm px-8 text-center">
                <BrainCircuit
                  size={28}
                  strokeWidth={1.2}
                  className="mx-auto text-white/20"
                />

                <p className="mt-5 text-sm text-white/35">
                  Run an analysis to inspect model outputs.
                </p>

                <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.16em] text-white/20">
                  SwahBERT / AfroXLM-R research environment
                </p>
              </div>
            </div>
          )}

          {analyseResult && (
            <section className="border border-white/10">
              <div className="border-b border-white/10 p-5">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-emerald-300/60">
                      Representation
                    </p>
                    <h2 className="mt-2 text-lg">
                      Encoder output
                    </h2>
                  </div>

                  <div className="text-right font-mono text-[10px] text-white/30">
                    {analyseResult.device}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 border-b border-white/10 sm:grid-cols-4">
                <Metric
                  label="Tokens"
                  value={analyseResult.token_count}
                />
                <Metric
                  label="Dimensions"
                  value={analyseResult.embedding_dimension}
                />
                <Metric
                  label="L2 norm"
                  value={analyseResult.embedding_norm.toFixed(3)}
                />
                <Metric
                  label="Latency"
                  value={`${analyseResult.inference_time_ms.toFixed(0)} ms`}
                />
              </div>

              <div className="p-5">
                <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                  Model
                </p>

                <p className="text-sm text-white/65">
                  {analyseResult.model}
                </p>

                <p className="mt-7 mb-3 font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                  Embedding preview
                </p>

                <div className="grid grid-cols-4 gap-px overflow-hidden border border-white/10 bg-white/10 sm:grid-cols-8">
                  {analyseResult.sentence_embedding
                    .slice(0, 32)
                    .map((value, index) => (
                      <div
                        key={index}
                        className="bg-[#0b0d10] p-2 text-center font-mono text-[9px] text-white/35"
                      >
                        {value.toFixed(3)}
                      </div>
                    ))}
                </div>

                <p className="mt-4 text-xs leading-5 text-white/30">
                  Showing the first 32 of{" "}
                  {analyseResult.embedding_dimension} dimensions.
                </p>
              </div>
            </section>
          )}

          {posResult && (
            <section className="border border-white/10">
              <div className="border-b border-white/10 p-5">
                <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-emerald-300/60">
                  Linguistic analysis
                </p>

                <h2 className="mt-2 text-lg">
                  Part-of-speech predictions
                </h2>
              </div>

              <div className="p-5">
                <div className="flex flex-wrap gap-2">
                  {posResult.tokens.map((item, index) => (
                    <div
                      key={`${item.token}-${index}`}
                      className="border border-white/10 bg-white/[0.025] px-3 py-2"
                    >
                      <div className="text-sm">
                        {item.token}
                      </div>

                      <div className="mt-1 flex gap-2 font-mono text-[9px]">
                        <span className="text-emerald-300/70">
                          {item.tag}
                        </span>
                        <span className="text-white/25">
                          {(item.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-6 border-t border-white/10 pt-4 text-xs leading-5 text-white/30">
                  Confidence is the model&apos;s softmax score and should not
                  be interpreted as calibrated probability of correctness.
                </div>
              </div>
            </section>
          )}
        </div>
      </section>
    </main>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="border-r border-white/10 p-4 last:border-r-0">
      <p className="font-mono text-[9px] uppercase tracking-[0.16em] text-white/25">
        {label}
      </p>

      <p className="mt-2 text-sm text-white/70">
        {value}
      </p>
    </div>
  );
}