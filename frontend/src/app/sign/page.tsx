"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Braces,
  Hand,
  Loader2,
  Scan,
  Upload,
  Video,
} from "lucide-react";
import LandmarkSkeleton from "@/components/sign/LandmarkSkeleton";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type LandmarkFrame = {
  frame_index: number;
  timestamp_seconds: number;
  landmarks: number[];
};

type LandmarkExtractionResult = {
  filename: string;
  frames_read: number;
  frames_detected: number;
  feature_dimension: number;
  duration_seconds: number;
  detection_rate: number;
  frames: LandmarkFrame[];
  note?: string | null;
};

type KSLPrediction = {
  index: number;
  label: string;
  probability: number;
};

type KSLAdapterDiagnostics = {
  native_shape: number[];
  hands_shape: number[];
  checkpoint_shape: number[];
  zero_padding_verified: boolean;
  zero_padding_nonzero_count: number;
  temporal_strategy: string;
  note?: string | null;
};

type KSLReliabilitySignals = {
  hand_coverage: number;
  top_probability: number;
  second_probability: number;
  probability_margin: number;
};

type KSLReliabilityChecks = {
  hand_coverage: boolean;
  top_probability: boolean;
  probability_margin: boolean;
};

type KSLReliabilityThresholds = {
  minimum_hand_coverage: number;
  minimum_top_probability: number;
  minimum_probability_margin: number;
};

type KSLReliabilityAssessment = {
  decision: string;
  accepted: boolean;

  signals: KSLReliabilitySignals;
  checks: KSLReliabilityChecks;
  thresholds: KSLReliabilityThresholds;

  reasons: string[];
  note?: string | null;
};

type KSLClassificationResult = {
  filename: string;

  predicted_label: string;
  predicted_probability: number;

  predictions: KSLPrediction[];

  model: string;
  device: string;

  closed_set: boolean;
  supported_labels: string[];

  frames_read: number;
  frames_detected: number;
  detection_rate: number;

  adapter: KSLAdapterDiagnostics;

  reliability: KSLReliabilityAssessment;

  warning: string;
};

export default function SignLabPage() {
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [result, setResult] =
    useState<LandmarkExtractionResult | null>(null);

  const [classification, setClassification] =
    useState<KSLClassificationResult | null>(null);

  const [classifying, setClassifying] =
    useState(false);

  const [selectedFrameIndex, setSelectedFrameIndex] =
    useState(0);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setVideoUrl(null);
      return;
    }

    const url = URL.createObjectURL(file);
    setVideoUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  function selectFile(selected: File | null) {
    if (!selected) return;

    setFile(selected);
    setResult(null);
    setSelectedFrameIndex(0);
    setError(null);
    setClassification(null);
  }

  async function extractLandmarks() {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(
        `${API_URL}/sign/landmarks`,
        {
          method: "POST",
          body: form,
        }
      );

      if (!response.ok) {
        let message = `Landmark extraction failed (${response.status})`;

        try {
          const body = await response.json();

          if (body?.detail) {
            message = body.detail;
          }
        } catch {
          // Keep fallback message.
        }

        throw new Error(message);
      }

      const data: LandmarkExtractionResult =
        await response.json();

      setResult(data);
      setSelectedFrameIndex(0);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to extract KSL landmarks."
      );
    } finally {
      setLoading(false);
    }
  }

  async function classifySign() {
  if (!file) return;

  setClassifying(true);
  setError(null);
  setClassification(null);

  try {
    const form = new FormData();
    form.append("file", file);

    const response = await fetch(
      `${API_URL}/sign/classify`,
      {
        method: "POST",
        body: form,
      }
    );

    if (!response.ok) {
      let message =
        `KSL classification failed (${response.status})`;

      try {
        const body = await response.json();

        if (body?.detail) {
          message = body.detail;
        }
      } catch {
        // Preserve fallback.
      }

      throw new Error(message);
    }

    const data: KSLClassificationResult =
      await response.json();

    setClassification(data);
  } catch (err) {
    setError(
      err instanceof Error
        ? err.message
        : "Unable to run KSL recognition."
    );
  } finally {
    setClassifying(false);
  }
}

  const selectedFrame = useMemo(() => {
    if (!result?.frames.length) return null;

    return (
      result.frames[selectedFrameIndex] ??
      result.frames[0]
    );
  }, [result, selectedFrameIndex]);

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
            <Hand
              size={17}
              className="text-emerald-300/70"
            />

            <span className="font-mono text-xs uppercase tracking-[0.2em] text-white/45">
              LughaSign
            </span>
          </div>
        </div>
      </header>

      <section className="border-b border-white/10">
        <div className="mx-auto max-w-7xl px-6 py-16 lg:px-10 lg:py-20">
          <p className="mb-4 font-mono text-xs uppercase tracking-[0.25em] text-emerald-300/65">
            Sign / 03
          </p>

          <h1 className="max-w-5xl text-5xl tracking-[-0.045em] sm:text-6xl">
            Kenyan Sign Language
            <span className="block text-white/30 sm:inline sm:ml-3">
              Representation Laboratory
            </span>
          </h1>

          <p className="mt-7 max-w-3xl text-sm leading-7 text-white/45">
            Convert sign-language video into structured body and hand
            landmarks. Inspect detection coverage and the resulting
            spatiotemporal representation before any classifier is
            allowed to make a prediction.
          </p>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-8 px-6 py-12 lg:grid-cols-[0.85fr_1.15fr] lg:px-10">
        <div className="space-y-6">
          <section className="border border-white/10 p-6">
            <div className="mb-6 flex items-center gap-3">
              <Video
                size={17}
                className="text-white/45"
              />

              <h2 className="text-sm font-medium">
                Video sample
              </h2>
            </div>

            <input
              ref={inputRef}
              type="file"
              accept="video/*,.mp4,.mov,.m4v,.webm"
              className="hidden"
              onChange={(event) =>
                selectFile(
                  event.target.files?.[0] ?? null
                )
              }
            />

            {!videoUrl ? (
              <button
                type="button"
                onClick={() =>
                  inputRef.current?.click()
                }
                className="flex min-h-56 w-full flex-col items-center justify-center border border-dashed border-white/15 bg-white/[0.015] transition hover:border-white/30 hover:bg-white/[0.025]"
              >
                <Upload
                  size={25}
                  strokeWidth={1.3}
                  className="text-white/30"
                />

                <span className="mt-4 text-sm text-white/45">
                  Select a sign-language video
                </span>

                <span className="mt-2 font-mono text-[10px] uppercase tracking-[0.16em] text-white/20">
                  MP4 / MOV / WEBM
                </span>
              </button>
            ) : (
              <div>
                <video
                  src={videoUrl}
                  controls
                  className="aspect-video w-full bg-black object-contain"
                />

                <div className="mt-4 flex items-center justify-between gap-4">
                  <div className="min-w-0">
                    <p className="truncate text-sm text-white/65">
                      {file?.name}
                    </p>

                    {file && (
                      <p className="mt-1 font-mono text-[10px] text-white/25">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    )}
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      inputRef.current?.click()
                    }
                    className="shrink-0 border border-white/10 px-3 py-2 text-[10px] text-white/40 transition hover:border-white/25 hover:text-white"
                  >
                    Change video
                  </button>
                </div>
              </div>
            )}

            <button
              onClick={extractLandmarks}
              disabled={!file || loading}
              className="mt-5 flex w-full items-center justify-center gap-2 bg-[#f1efe8] px-4 py-3 text-xs font-medium text-[#080a0d] transition hover:bg-white disabled:opacity-35"
            >
              {loading ? (
                <Loader2
                  size={14}
                  className="animate-spin"
                />
              ) : (
                <Scan size={14} />
              )}

              {loading
                ? "Extracting landmarks..."
                : "Extract landmarks"}
            </button>

            <button
                onClick={classifySign}
                disabled={!file || loading || classifying}
                className="mt-3 flex w-full items-center justify-center gap-2 border border-emerald-300/20 px-4 py-3 text-xs text-emerald-200/70 transition hover:border-emerald-300/40 hover:text-emerald-100 disabled:opacity-35"
                >
                {classifying ? (
                    <Loader2
                    size={14}
                    className="animate-spin"
                    />
                ) : (
                    <Hand size={14} />
                )}

                {classifying
                    ? "Running experimental classifier..."
                    : "Run 4-gloss classifier"}
            </button>

            {error && (
              <div className="mt-5 border border-red-400/20 bg-red-400/[0.05] p-4 text-xs leading-5 text-red-200/75">
                {error}
              </div>
            )}
          </section>

          <section className="border border-white/10 p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/30">
              Representation
            </p>

            <div className="mt-5 space-y-4 text-sm leading-6 text-white/40">
              <p>
                Each frame is represented using MediaPipe Holistic
                pose and hand coordinates.
              </p>

              <div className="border-l border-emerald-300/30 py-2 pl-4 font-mono text-xs leading-6 text-white/60">
                xₜ ∈ ℝ²²⁵
                <br />
                X ∈ ℝᵀˣ²²⁵
              </div>

              <div className="space-y-2 font-mono text-[10px] text-white/35">
                <RepresentationRow
                  label="Pose"
                  value="33 × 3 = 99"
                />

                <RepresentationRow
                  label="Left hand"
                  value="21 × 3 = 63"
                />

                <RepresentationRow
                  label="Right hand"
                  value="21 × 3 = 63"
                />

                <RepresentationRow
                  label="Total"
                  value="225 features / frame"
                  strong
                />
              </div>

              <p>
                Missing landmarks are zero-filled in the extraction
                representation. Detection failure can therefore
                propagate into downstream recognition.
              </p>
            </div>
          </section>
        </div>

        <div>
          {!result && (
            <div className="flex min-h-[600px] items-center justify-center border border-dashed border-white/10">
              <div className="max-w-sm px-8 text-center">
                <Hand
                  size={31}
                  strokeWidth={1.1}
                  className="mx-auto text-white/20"
                />

                <p className="mt-5 text-sm text-white/35">
                  Upload a video to inspect its landmark
                  representation.
                </p>

                <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.16em] text-white/20">
                  Video → Pose + Hands → 225D
                </p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-6">
              <section className="border border-white/10">
                <div className="border-b border-white/10 p-5">
                  <div className="flex items-start justify-between gap-5">
                    <div>
                      <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-emerald-300/60">
                        Extraction
                      </p>

                      <h2 className="mt-2 text-lg">
                        Landmark diagnostics
                      </h2>
                    </div>

                    <p className="max-w-52 truncate font-mono text-[9px] text-white/25">
                      {result.filename}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 border-b border-white/10 sm:grid-cols-4">
                  <Metric
                    label="Frames"
                    value={result.frames_read}
                  />

                  <Metric
                    label="Detected"
                    value={result.frames_detected}
                  />

                  <Metric
                    label="Features"
                    value={result.feature_dimension}
                  />

                  <Metric
                    label="Duration"
                    value={`${result.duration_seconds.toFixed(
                      2
                    )} s`}
                  />
                </div>

                <div className="p-6">
                  <div className="flex items-end justify-between gap-6">
                    <div>
                      <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                        Hand detection coverage
                      </p>

                      <p className="mt-3 text-4xl tracking-[-0.04em]">
                        {(result.detection_rate * 100).toFixed(
                          1
                        )}
                        <span className="ml-1 text-xl text-white/30">
                          %
                        </span>
                      </p>
                    </div>

                    <p className="max-w-xs text-right text-xs leading-5 text-white/30">
                      Fraction of processed frames containing detected
                      hand landmarks.
                    </p>
                  </div>

                  <div className="mt-5 h-1.5 overflow-hidden bg-white/10">
                    <div
                      className="h-full bg-emerald-300/70 transition-all duration-500"
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(
                            0,
                            result.detection_rate * 100
                          )
                        )}%`,
                      }}
                    />
                  </div>

                  <div className="mt-3 flex justify-between font-mono text-[9px] text-white/20">
                    <span>0%</span>
                    <span>100%</span>
                  </div>
                </div>
              </section>

              {classification && (
                <section className="border border-white/10">
                    <div className="border-b border-white/10 p-5">
                    <div className="flex flex-wrap items-start justify-between gap-5">
                        <div>
                        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-emerald-300/60">
                            Experimental recognition
                        </p>

                        <h2 className="mt-2 text-lg">
                            4-gloss KSL baseline
                        </h2>
                        </div>

                        <div className="border border-amber-300/20 px-3 py-2 font-mono text-[9px] uppercase tracking-[0.14em] text-amber-200/60">
                        Closed set
                        </div>
                    </div>
                    </div>

                    <div className="p-6">
                    <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                        Model output
                    </p>

                    <div
                    className={`border-b p-6 ${
                        classification.reliability.accepted
                        ? "border-emerald-300/20 bg-emerald-300/[0.025]"
                        : "border-amber-300/20 bg-amber-300/[0.025]"
                    }`}
                    >
                    <div className="flex flex-wrap items-center justify-between gap-5">
                        <div>
                        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/25">
                            System decision
                        </p>

                        <p
                            className={`mt-2 text-2xl tracking-[-0.03em] ${
                            classification.reliability.accepted
                                ? "text-emerald-200"
                                : "text-amber-200"
                            }`}
                        >
                            {classification.reliability.accepted
                            ? "Provisionally accepted"
                            : "Unreliable input"}
                        </p>
                        </div>

                        <div
                        className={`h-3 w-3 rounded-full ${
                            classification.reliability.accepted
                            ? "bg-emerald-300"
                            : "bg-amber-300"
                        }`}
                        />
                    </div>

                    {!classification.reliability.accepted && (
                        <p className="mt-4 max-w-2xl text-xs leading-6 text-amber-100/45">
                        The classifier still produced a closed-set model
                        preference, but LughaLab does not consider the
                        available evidence sufficient to interpret that
                        output as a reliable KSL recognition.
                        </p>
                    )}
                    </div>

                    <div className="mt-3 flex items-end gap-4">
                        <p className="text-5xl tracking-[-0.05em]">
                        {classification.predicted_label}
                        </p>

                        <p className="pb-1 font-mono text-xs text-white/30">
                        {(
                            classification.predicted_probability *
                            100
                        ).toFixed(1)}
                        % softmax
                        </p>
                    </div>

                    <p className="mt-4 max-w-xl text-xs leading-5 text-white/30">
                        Highest-scoring class among four supported
                        labels. Softmax probability is not a calibrated
                        estimate that the gesture actually represents
                        this gloss.
                    </p>
                    </div>

                    <div className="border-t border-white/10 p-6">
                    <p className="mb-5 font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                        Class distribution
                    </p>

                    <div className="space-y-4">
                        {classification.predictions.map(
                        (prediction) => (
                            <ProbabilityBar
                            key={prediction.label}
                            label={prediction.label}
                            probability={
                                prediction.probability
                            }
                            />
                        )
                        )}
                    </div>
                    </div>

                    <div className="border-t border-white/10 p-6">
                        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                            Reliability gate
                        </p>

                        <div className="mt-5 space-y-3">
                            <ReliabilityCheck
                            label="Hand landmark coverage"
                            value={`${(
                                classification.reliability.signals
                                .hand_coverage * 100
                            ).toFixed(1)}%`}
                            threshold={`≥ ${(
                                classification.reliability.thresholds
                                .minimum_hand_coverage * 100
                            ).toFixed(0)}%`}
                            passed={
                                classification.reliability.checks
                                .hand_coverage
                            }
                            />

                            <ReliabilityCheck
                            label="Softmax peak"
                            value={`${(
                                classification.reliability.signals
                                .top_probability * 100
                            ).toFixed(1)}%`}
                            threshold={`≥ ${(
                                classification.reliability.thresholds
                                .minimum_top_probability * 100
                            ).toFixed(0)}%`}
                            passed={
                                classification.reliability.checks
                                .top_probability
                            }
                            />

                            <ReliabilityCheck
                            label="Top-two margin"
                            value={`${(
                                classification.reliability.signals
                                .probability_margin * 100
                            ).toFixed(1)} pp`}
                            threshold={`≥ ${(
                                classification.reliability.thresholds
                                .minimum_probability_margin * 100
                            ).toFixed(0)} pp`}
                            passed={
                                classification.reliability.checks
                                .probability_margin
                            }
                            />
                        </div>
                        </div>

                        {classification.reliability.reasons.length > 0 && (
                        <div className="mt-5 border border-amber-300/15 p-4">
                            <p className="font-mono text-[9px] uppercase tracking-[0.16em] text-amber-200/45">
                            Rejection reason
                            </p>

                            <div className="mt-3 space-y-2">
                            {classification.reliability.reasons.map(
                                (reason) => (
                                <p
                                    key={reason}
                                    className="font-mono text-[10px] text-amber-100/50"
                                >
                                    {formatReliabilityReason(reason)}
                                </p>
                                )
                            )}
                            </div>
                        </div>
                        )}

                    <div className="grid grid-cols-2 border-t border-white/10 sm:grid-cols-4">
                    <Metric
                        label="Frames"
                        value={classification.frames_read}
                    />

                    <Metric
                        label="Hands detected"
                        value={classification.frames_detected}
                    />

                    <Metric
                        label="Coverage"
                        value={`${(
                        classification.detection_rate * 100
                        ).toFixed(1)}%`}
                    />

                    <Metric
                        label="Device"
                        value={classification.device}
                    />
                    </div>

                    <div className="border-t border-white/10 p-6">
                    <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                        Checkpoint adapter
                    </p>

                    <div className="mt-5 grid gap-px bg-white/10 sm:grid-cols-2">
                        <AdapterMetric
                        label="Native"
                        value={formatShape(
                            classification.adapter.native_shape
                        )}
                        />

                        <AdapterMetric
                        label="Hands"
                        value={formatShape(
                            classification.adapter.hands_shape
                        )}
                        />

                        <AdapterMetric
                        label="Checkpoint"
                        value={formatShape(
                            classification.adapter.checkpoint_shape
                        )}
                        />

                        <AdapterMetric
                        label="Temporal"
                        value={
                            classification.adapter.temporal_strategy
                        }
                        />
                    </div>

                    <div className="mt-4 flex items-center justify-between border border-white/[0.07] p-3">
                        <span className="font-mono text-[9px] uppercase tracking-[0.14em] text-white/25">
                        99D compatibility padding
                        </span>

                        <span
                        className={`font-mono text-[10px] ${
                            classification.adapter
                            .zero_padding_verified
                            ? "text-emerald-300/65"
                            : "text-red-300/70"
                        }`}
                        >
                        {classification.adapter
                            .zero_padding_verified
                            ? "verified"
                            : "failed"}
                        </span>
                    </div>
                    </div>

                    <div className="border-t border-amber-300/15 bg-amber-300/[0.025] p-6">
                    <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-amber-200/55">
                        Interpretation warning
                    </p>

                    <p className="mt-3 text-xs leading-6 text-amber-100/40">
                        {classification.warning}
                    </p>
                    </div>
                </section>
                )}

              <section className="border border-white/10">
                <div className="border-b border-white/10 p-5">
                  <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-emerald-300/60">
                    Temporal representation
                  </p>

                  <h2 className="mt-2 text-lg">
                    Frame explorer
                  </h2>
                </div>

                {result.frames.length > 0 ? (
                  <>
                    <div className="p-5">
                      <input
                        type="range"
                        min={0}
                        max={Math.max(
                          0,
                          result.frames.length - 1
                        )}
                        value={selectedFrameIndex}
                        onChange={(event) =>
                          setSelectedFrameIndex(
                            Number(event.target.value)
                          )
                        }
                        className="w-full accent-emerald-300"
                      />

                      <div className="mt-3 flex justify-between font-mono text-[9px] text-white/25">
                        <span>
                          frame{" "}
                          {selectedFrame?.frame_index ?? 0}
                        </span>

                        <span>
                          {selectedFrame
                            ? `${selectedFrame.timestamp_seconds.toFixed(
                                3
                              )} s`
                            : "—"}
                        </span>
                      </div>
                    </div>

                    {selectedFrame && (
                      <div className="border-t border-white/10 p-5">
                        <div className="mb-7">
                        <div className="mb-4 flex items-center justify-between gap-4">
                            <div>
                            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                                Spatial reconstruction
                            </p>

                            <p className="mt-2 text-xs text-white/30">
                                Pose and hand coordinates reconstructed from the
                                selected 225-dimensional frame.
                            </p>
                            </div>

                            <div className="text-right font-mono text-[9px] text-white/20">
                            t = {selectedFrame.timestamp_seconds.toFixed(3)} s
                            </div>
                        </div>

                        <LandmarkSkeleton
                            landmarks={selectedFrame.landmarks}
                        />
                        </div>
                        <div className="mb-4 flex items-center gap-2">
                          <Braces
                            size={14}
                            className="text-white/35"
                          />

                          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                            Feature vector preview
                          </p>
                        </div>

                        <div className="grid grid-cols-4 gap-px overflow-hidden border border-white/10 bg-white/10 sm:grid-cols-6 lg:grid-cols-9">
                          {selectedFrame.landmarks
                            .slice(0, 36)
                            .map((value, index) => (
                              <div
                                key={index}
                                className="bg-[#0b0d10] p-2"
                              >
                                <div className="font-mono text-[8px] text-white/15">
                                  {index
                                    .toString()
                                    .padStart(3, "0")}
                                </div>

                                <div className="mt-1 truncate font-mono text-[9px] text-white/40">
                                  {value.toFixed(3)}
                                </div>
                              </div>
                            ))}
                        </div>

                        <p className="mt-4 text-xs leading-5 text-white/25">
                          Showing dimensions 0–35 of{" "}
                          {selectedFrame.landmarks.length}. The
                          complete frame representation is retained by
                          the API.
                        </p>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="p-8 text-sm text-white/30">
                    No landmark frames were returned.
                  </div>
                )}
              </section>

              {result.note && (
                <section className="border border-white/10 p-5">
                  <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/25">
                    Pipeline note
                  </p>

                  <p className="mt-3 text-xs leading-6 text-white/35">
                    {result.note}
                  </p>
                </section>
              )}
            </div>
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

function RepresentationRow({
  label,
  value,
  strong = false,
}: {
  label: string;
  value: string;
  strong?: boolean;
}) {
  return (
    <div
      className={`flex justify-between border-b border-white/[0.06] py-2 ${
        strong
          ? "text-emerald-300/60"
          : "text-white/35"
      }`}
    >
      <span>{label}</span>
      <span>{value}</span>
    </div>
  );
}

function ProbabilityBar({
  label,
  probability,
}: {
  label: string;
  probability: number;
}) {
  const percentage = Math.max(
    0,
    Math.min(100, probability * 100)
  );

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs text-white/55">
          {label}
        </span>

        <span className="font-mono text-[10px] text-white/30">
          {percentage.toFixed(1)}%
        </span>
      </div>

      <div className="h-1.5 overflow-hidden bg-white/[0.07]">
        <div
          className="h-full bg-emerald-300/60"
          style={{
            width: `${percentage}%`,
          }}
        />
      </div>
    </div>
  );
}

function AdapterMetric({
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

      <p className="mt-2 break-all font-mono text-[10px] text-white/50">
        {value}
      </p>
    </div>
  );
}

function formatShape(
  shape: number[]
) {
  return `(${shape.join(", ")})`;
}

function ReliabilityCheck({
  label,
  value,
  threshold,
  passed,
}: {
  label: string;
  value: string;
  threshold: string;
  passed: boolean;
}) {
  return (
    <div className="grid grid-cols-[1fr_auto_auto] items-center gap-4 border border-white/[0.07] p-3">
      <div>
        <p className="text-xs text-white/55">
          {label}
        </p>

        <p className="mt-1 font-mono text-[9px] text-white/20">
          threshold {threshold}
        </p>
      </div>

      <span className="font-mono text-[10px] text-white/45">
        {value}
      </span>

      <span
        className={`min-w-12 text-right font-mono text-[9px] uppercase ${
          passed
            ? "text-emerald-300/70"
            : "text-amber-300/70"
        }`}
      >
        {passed ? "pass" : "fail"}
      </span>
    </div>
  );
}

function formatReliabilityReason(
  reason: string
) {
  const labels: Record<string, string> = {
    insufficient_hand_landmark_coverage:
      "Insufficient hand-landmark coverage",

    weak_closed_set_softmax_peak:
      "Weak closed-set softmax peak",

    ambiguous_closed_set_prediction:
      "Ambiguous top-two class prediction",
  };

  return labels[reason] ?? reason;
}