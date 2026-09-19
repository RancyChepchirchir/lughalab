"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  AudioLines,
  FileAudio,
  Loader2,
  Upload,
  Waves,
} from "lucide-react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type Segment = {
  start: number;
  end: number;
  text: string;
};

type TranscriptionResult = {
  model: string;
  language: string;
  device: string;
  sampling_rate: number;
  duration_seconds: number;
  transcription: string;
  inference_time_ms: number;
  segments: Segment[];
  note?: string | null;
};

const models = [
  {
    id: "whisper-small",
    name: "Whisper Small",
    type: "Zero-shot baseline",
    description:
      "General multilingual speech model evaluated without Swahili-specific adaptation.",
  },
  {
    id: "sauti-v1",
    name: "Sauti ASR v1",
    type: "Adapted model",
    description:
      "Whisper-medium model adapted specifically for Swahili automatic speech recognition.",
  },
];

export default function SpeechLabPage() {
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState("whisper-small");
  const [result, setResult] =
    useState<TranscriptionResult | null>(null);
  const [comparisonResults, setComparisonResults] = useState<
    Record<string, TranscriptionResult>
    >({});  

  const [loadingComparison, setLoadingComparison] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function selectFile(selected: File | null) {
    if (!selected) return;

    setFile(selected);
    setResult(null);
    setComparisonResults({});
    setError(null);
  }

  async function transcribe() {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const form = new FormData();

      form.append("file", file);
      form.append("model", model);

      const response = await fetch(
        `${API_URL}/speech/transcribe`,
        {
          method: "POST",
          body: form,
        }
      );

      if (!response.ok) {
        const message = await response.text();

        throw new Error(
          message ||
            `Transcription failed (${response.status})`
        );
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to transcribe the recording."
      );
    } finally {
      setLoading(false);
    }
  }

  async function compareModels() {
  if (!file) return;

  setLoadingComparison(true);
  setError(null);
  setComparisonResults({});

  try {
    const outputs: Record<string, TranscriptionResult> = {};

    for (const modelId of ["whisper-small", "sauti-v1"]) {
      const form = new FormData();

      form.append("file", file);
      form.append("model", modelId);

      const response = await fetch(
        `${API_URL}/speech/transcribe`,
        {
          method: "POST",
          body: form,
        }
      );

      if (!response.ok) {
        const message = await response.text();

        throw new Error(
          message ||
            `${modelId} transcription failed (${response.status})`
        );
      }

      const data: TranscriptionResult =
        await response.json();

      outputs[modelId] = data;

      // Update after each model so the first result appears
      // while the second model is still running.
      setComparisonResults({ ...outputs });
    }
  } catch (err) {
    setError(
      err instanceof Error
        ? err.message
        : "Unable to compare the ASR models."
    );
  } finally {
    setLoadingComparison(false);
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
            <AudioLines
              size={17}
              className="text-emerald-300/70"
            />

            <span className="font-mono text-xs uppercase tracking-[0.2em] text-white/45">
              LughaSpeech
            </span>
          </div>
        </div>
      </header>

      <section className="border-b border-white/10">
        <div className="mx-auto max-w-7xl px-6 py-16 lg:px-10 lg:py-20">
          <p className="mb-4 font-mono text-xs uppercase tracking-[0.25em] text-emerald-300/65">
            Speech / 02
          </p>

          <h1 className="max-w-5xl text-5xl tracking-[-0.045em] sm:text-6xl">
            Swahili
            <span className="ml-3 text-white/30">
              Speech Laboratory
            </span>
          </h1>

          <p className="mt-7 max-w-2xl text-sm leading-7 text-white/45">
            Compare zero-shot multilingual speech recognition with
            Swahili-adapted ASR. Inspect transcripts, inference
            efficiency and model behaviour rather than treating WER as
            the entire story.
          </p>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-8 px-6 py-12 lg:grid-cols-[0.9fr_1.1fr] lg:px-10">
        <div className="space-y-6">
          <section className="border border-white/10 p-6">
            <div className="mb-6 flex items-center gap-3">
              <FileAudio size={17} className="text-white/45" />
              <h2 className="text-sm font-medium">
                Audio sample
              </h2>
            </div>

            <input
              ref={inputRef}
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.flac"
              className="hidden"
              onChange={(event) =>
                selectFile(
                  event.target.files?.[0] ?? null
                )
              }
            />

            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="flex min-h-40 w-full flex-col items-center justify-center border border-dashed border-white/15 bg-white/[0.015] transition hover:border-white/30 hover:bg-white/[0.025]"
            >
              <Upload
                size={24}
                strokeWidth={1.3}
                className="text-white/30"
              />

              {file ? (
                <>
                  <span className="mt-4 text-sm text-white/70">
                    {file.name}
                  </span>

                  <span className="mt-1 font-mono text-[10px] text-white/25">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </>
              ) : (
                <>
                  <span className="mt-4 text-sm text-white/45">
                    Select an audio recording
                  </span>

                  <span className="mt-2 font-mono text-[10px] uppercase tracking-[0.16em] text-white/20">
                    WAV / MP3 / M4A / FLAC
                  </span>
                </>
              )}
            </button>

            {file && (
              <audio
                className="mt-5 w-full"
                controls
                src={URL.createObjectURL(file)}
              />
            )}
          </section>

          <section className="border border-white/10 p-6">
            <p className="mb-5 font-mono text-[10px] uppercase tracking-[0.2em] text-white/30">
              Model
            </p>

            <div className="space-y-3">
              {models.map((item) => {
                const selected = model === item.id;

                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      setModel(item.id);
                      setResult(null);
                      setComparisonResults({});
                    }}
                    className={`w-full border p-4 text-left transition ${
                      selected
                        ? "border-emerald-300/35 bg-emerald-300/[0.04]"
                        : "border-white/10 hover:border-white/20"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="text-sm">
                          {item.name}
                        </div>

                        <div className="mt-1 font-mono text-[9px] uppercase tracking-[0.15em] text-emerald-300/55">
                          {item.type}
                        </div>
                      </div>

                      <div
                        className={`mt-1 h-2 w-2 rounded-full ${
                          selected
                            ? "bg-emerald-300"
                            : "bg-white/15"
                        }`}
                      />
                    </div>

                    <p className="mt-3 text-xs leading-5 text-white/35">
                      {item.description}
                    </p>
                  </button>
                );
              })}
            </div>

            <button
              onClick={transcribe}
              disabled={!file || loading}
              className="mt-5 flex w-full items-center justify-center gap-2 bg-[#f1efe8] px-4 py-3 text-xs font-medium text-[#080a0d] transition hover:bg-white disabled:opacity-35"
            >
              {loading ? (
                <Loader2
                  size={14}
                  className="animate-spin"
                />
              ) : (
                <Waves size={14} />
              )}

              {loading
                ? "Transcribing..."
                : "Run transcription"}
            </button>

            <button
            onClick={compareModels}
            disabled={!file || loading || loadingComparison}
            className="mt-3 flex w-full items-center justify-center gap-2 border border-white/15 px-4 py-3 text-xs text-white/65 transition hover:border-emerald-300/30 hover:text-white disabled:opacity-35"
            >
            {loadingComparison ? (
                <Loader2 size={14} className="animate-spin" />
            ) : (
                <AudioLines size={14} />
            )}

            {loadingComparison
                ? "Comparing models..."
                : "Compare Whisper vs Sauti"}
            </button>

            {error && (
              <div className="mt-5 border border-red-400/20 bg-red-400/[0.05] p-4 text-xs leading-5 text-red-200/75">
                {error}
              </div>
            )}
          </section>

          <section className="border border-white/10 p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/30">
              Research note
            </p>

            <div className="mt-5 space-y-4 text-sm leading-6 text-white/40">
              <p>
                Automatic speech recognition estimates the most
                probable text sequence conditioned on the acoustic
                signal.
              </p>

              <div className="overflow-x-auto border-l border-emerald-300/30 py-2 pl-4 font-mono text-xs text-white/60">
                ŷ = argmaxᵧ Pθ(y | x, ℓ = Swahili)
              </div>

              <p>
                Orthographic disagreement does not necessarily imply
                semantic failure, particularly for informal speech,
                names and code-switching.
              </p>
            </div>
          </section>
        </div>

        <div>
          {!result && (
            <div className="flex min-h-[520px] items-center justify-center border border-dashed border-white/10">
              <div className="max-w-sm px-8 text-center">
                <AudioLines
                  size={30}
                  strokeWidth={1.1}
                  className="mx-auto text-white/20"
                />

                <p className="mt-5 text-sm text-white/35">
                  Upload a Swahili recording and run a model.
                </p>

                <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.16em] text-white/20">
                  Zero-shot / Adapted ASR
                </p>
              </div>
            </div>
          )}

          {result && (
            <section className="border border-white/10">
              <div className="border-b border-white/10 p-5">
                <div className="flex items-start justify-between gap-6">
                  <div>
                    <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-emerald-300/60">
                      Transcription
                    </p>

                    <h2 className="mt-2 text-lg">
                      Model output
                    </h2>
                  </div>

                  <div className="text-right">
                    <p className="font-mono text-[10px] text-white/30">
                      {result.device}
                    </p>

                    <p className="mt-1 font-mono text-[9px] text-white/20">
                      {result.model}
                    </p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 border-b border-white/10 sm:grid-cols-4">
                <Metric
                    label="Duration"
                    value={`${result.duration_seconds.toFixed(2)} s`}
                />

                <Metric
                    label="Inference"
                    value={`${result.inference_time_ms.toFixed(0)} ms`}
                />

                <Metric
                    label="RTF"
                    value={
                    result.duration_seconds > 0
                        ? (
                            result.inference_time_ms /
                            1000 /
                            result.duration_seconds
                        ).toFixed(3)
                        : "—"
                    }
                />

                <Metric
                    label="Sample rate"
                    value={`${(result.sampling_rate / 1000).toFixed(0)} kHz`}
                />
                </div>

              <div className="p-6">
                <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                  Transcript
                </p>

                <blockquote className="mt-5 border-l border-emerald-300/30 pl-5 text-xl leading-9 tracking-[-0.015em] text-white/80">
                  {result.transcription || "No speech decoded."}
                </blockquote>
              </div>

              {result.segments?.length > 0 && (
                <div className="border-t border-white/10 p-6">
                  <p className="mb-5 font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                    Temporal segments
                  </p>

                  <div className="space-y-px bg-white/10">
                    {result.segments.map(
                      (segment, index) => (
                        <div
                          key={`${segment.start}-${index}`}
                          className="grid gap-3 bg-[#0b0d10] p-4 sm:grid-cols-[100px_1fr]"
                        >
                          <span className="font-mono text-[10px] text-emerald-300/55">
                            {segment.start.toFixed(2)}–
                            {segment.end.toFixed(2)}s
                          </span>

                          <span className="text-sm text-white/55">
                            {segment.text}
                          </span>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}

              <div className="border-t border-white/10 p-5 text-xs leading-5 text-white/30">
                Real-time factor is inference time divided by audio
                duration. Values below 1 indicate processing faster
                than real time.
              </div>
            </section>
          )}

          {Object.keys(comparisonResults).length > 0 && (
            <section className="mt-6 border border-white/10">
                <div className="border-b border-white/10 p-5">
                <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-emerald-300/60">
                    Controlled comparison
                </p>

                <h2 className="mt-2 text-lg">
                    Zero-shot vs adapted ASR
                </h2>

                <p className="mt-3 max-w-xl text-xs leading-5 text-white/30">
                    Both systems receive the same audio recording. Transcript
                    differences therefore reflect model behaviour under a shared
                    acoustic input.
                </p>
                </div>

                <div className="grid lg:grid-cols-2">
                {["whisper-small", "sauti-v1"].map((modelId) => {
                    const output = comparisonResults[modelId];

                    const rtf =
                    output && output.duration_seconds > 0
                        ? output.inference_time_ms /
                        1000 /
                        output.duration_seconds
                        : null;

                    return (
                    <div
                        key={modelId}
                        className="min-h-80 border-b border-white/10 p-5 last:border-b-0 lg:border-b-0 lg:border-r lg:last:border-r-0"
                    >
                        <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-white/25">
                        {modelId === "whisper-small"
                            ? "Zero-shot baseline"
                            : "Swahili adapted"}
                        </p>

                        <h3 className="mt-2 text-sm text-white/70">
                        {modelId === "whisper-small"
                            ? "Whisper Small"
                            : "Sauti ASR v1"}
                        </h3>

                        {!output ? (
                        <div className="mt-10 flex items-center gap-2 text-xs text-white/25">
                            {loadingComparison && (
                            <Loader2
                                size={12}
                                className="animate-spin"
                            />
                            )}
                            Waiting for model...
                        </div>
                        ) : (
                        <>
                            <blockquote className="mt-7 min-h-32 border-l border-emerald-300/25 pl-4 text-sm leading-7 text-white/65">
                            {output.transcription ||
                                "No speech decoded."}
                            </blockquote>

                            <div className="mt-7 grid grid-cols-2 gap-px bg-white/10">
                            <ComparisonMetric
                                label="Inference"
                                value={`${output.inference_time_ms.toFixed(
                                0
                                )} ms`}
                            />

                            <ComparisonMetric
                                label="RTF"
                                value={
                                rtf !== null
                                    ? rtf.toFixed(3)
                                    : "—"
                                }
                            />

                            <ComparisonMetric
                                label="Duration"
                                value={`${output.duration_seconds.toFixed(
                                2
                                )} s`}
                            />

                            <ComparisonMetric
                                label="Sample rate"
                                value={`${(
                                output.sampling_rate / 1000
                                ).toFixed(0)} kHz`}
                            />
                            </div>
                        </>
                        )}
                    </div>
                    );
                })}
                </div>

                <div className="border-t border-white/10 p-5">
                <p className="text-xs leading-5 text-white/30">
                    This interactive comparison does not calculate WER or CER
                    because no reference transcript has been supplied. A more
                    plausible prediction is not automatically the lower-error
                    prediction.
                </p>
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

function ComparisonMetric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="bg-[#0b0d10] p-3">
      <p className="font-mono text-[9px] uppercase tracking-[0.14em] text-white/20">
        {label}
      </p>

      <p className="mt-2 font-mono text-xs text-white/55">
        {value}
      </p>
    </div>
  );
}