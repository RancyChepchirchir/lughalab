import {
  ArrowRight,
  AudioLines,
  BrainCircuit,
  Hand,
  Languages,
} from "lucide-react";
import Link from "next/link";

const tracks = [
  {
    index: "01",
    name: "LughaNLP",
    title: "African Language Intelligence",
    description:
      "Representation learning and linguistic evaluation for low-resource African languages, beginning with Swahili.",
    tags: ["SwahBERT", "AfroXLM-R", "POS", "Semantic Stress Tests"],
    icon: Languages,
    href: "/nlp",
  },
  {
    index: "02",
    name: "LughaSpeech",
    title: "Speech Intelligence",
    description:
      "Swahili automatic speech recognition, adaptation and error analysis across standard, informal and code-switched speech.",
    tags: ["Whisper", "Sauti-ASR", "WER", "Error Analysis"],
    icon: AudioLines,
    href: "/speech",
  },
  {
    index: "03",
    name: "LughaSign",
    title: "Sign Intelligence",
    description:
      "Landmark-based Kenyan Sign Language research spanning recognition, representation analysis and model reliability.",
    tags: ["KSL", "MediaPipe", "Transformer", "Attribution"],
    icon: Hand,
    href: "/sign",
  },
];

export default function Home() {
  return (
    <main className="min-h-screen bg-[#080a0d] text-[#f1efe8]">
      <section className="border-b border-white/10">
        <div className="mx-auto max-w-7xl px-6 pb-20 pt-8 lg:px-10 lg:pb-28">
          <nav className="mb-24 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center border border-white/20">
                <BrainCircuit size={18} />
              </div>
              <div>
                <div className="font-semibold tracking-tight">LughaLab</div>
                <div className="text-[10px] uppercase tracking-[0.24em] text-white/40">
                  Research Workstation
                </div>
              </div>
            </div>

            <div className="hidden gap-8 text-sm text-white/55 md:flex">
              <a className="transition hover:text-white" href="#tracks">
                Research
              </a>
              <a className="transition hover:text-white" href="#principles">
                Methodology
              </a>
              <a className="transition hover:text-white" href="#about">
                About
              </a>
            </div>
          </nav>

          <div className="max-w-5xl">
            <div className="mb-7 flex items-center gap-3 text-xs uppercase tracking-[0.28em] text-emerald-300/70">
              <span className="h-px w-8 bg-emerald-300/60" />
              African Language &amp; Sign Intelligence
            </div>

            <h1 className="max-w-4xl text-5xl font-medium leading-[0.98] tracking-[-0.045em] sm:text-7xl lg:text-[92px]">
              Language intelligence,
              <span className="block text-white/35">beyond the benchmark.</span>
            </h1>

            <p className="mt-9 max-w-2xl text-base leading-7 text-white/55 sm:text-lg">
              A multimodal research laboratory for studying how pretrained AI
              systems behave when adapted to low-resource African languages,
              speech and sign.
            </p>

            <div className="mt-12 flex flex-wrap gap-3">
              {["TEXT", "SPEECH", "SIGN", "EVALUATION"].map((item) => (
                <span
                  key={item}
                  className="border border-white/15 px-4 py-2 text-[11px] tracking-[0.2em] text-white/55"
                >
                  {item}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section id="tracks" className="border-b border-white/10">
        <div className="mx-auto max-w-7xl px-6 py-20 lg:px-10 lg:py-28">
          <div className="mb-14 grid gap-6 lg:grid-cols-2">
            <div>
              <p className="mb-3 text-xs uppercase tracking-[0.25em] text-white/35">
                01 / Research Tracks
              </p>
              <h2 className="text-3xl tracking-[-0.03em] sm:text-4xl">
                Three modalities.
                <span className="block text-white/35">One research question.</span>
              </h2>
            </div>

            <p className="max-w-xl self-end text-sm leading-6 text-white/45 lg:justify-self-end">
              How effectively can pretrained multilingual and
              modality-specific foundation models be adapted to low-resource
              East African languages, and where do they fail under language,
              domain, signer and modality shift?
            </p>
          </div>

          <div className="grid border-l border-t border-white/10 lg:grid-cols-3">
            {tracks.map((track) => {
              const Icon = track.icon;

              return (
                <Link
                  href={track.href}
                  key={track.name}
                  className="group block min-h-[420px] border-b border-r border-white/10 p-7 transition duration-300 hover:bg-white/[0.035]"
                >
                  <div className="flex items-start justify-between">
                    <span className="font-mono text-xs text-white/30">
                      {track.index}
                    </span>
                    <Icon
                      size={22}
                      strokeWidth={1.4}
                      className="text-white/45 transition group-hover:text-emerald-300"
                    />
                  </div>

                  <div className="mt-20">
                    <p className="mb-2 font-mono text-xs text-emerald-300/65">
                      {track.name}
                    </p>
                    <h3 className="text-2xl tracking-[-0.025em]">
                      {track.title}
                    </h3>
                    <p className="mt-5 max-w-sm text-sm leading-6 text-white/45">
                      {track.description}
                    </p>
                  </div>

                  <div className="mt-9 flex flex-wrap gap-2">
                    {track.tags.map((tag) => (
                      <span
                        key={tag}
                        className="border border-white/10 px-2.5 py-1.5 font-mono text-[10px] text-white/40"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>

                  <div className="mt-9 flex items-center gap-2 text-xs text-white/35 transition group-hover:text-white/70">
                    Explore research
                    <ArrowRight size={13} />
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      <section id="principles" className="border-b border-white/10">
        <div className="mx-auto grid max-w-7xl gap-14 px-6 py-20 lg:grid-cols-[0.8fr_1.2fr] lg:px-10 lg:py-28">
          <div>
            <p className="mb-3 text-xs uppercase tracking-[0.25em] text-white/35">
              02 / Research Philosophy
            </p>
            <h2 className="text-3xl tracking-[-0.03em]">
              Accuracy is only
              <span className="block text-white/35">the beginning.</span>
            </h2>
          </div>

          <div className="grid sm:grid-cols-2">
            {[
              [
                "Representation ≠ Understanding",
                "High cosine similarity does not by itself establish semantic understanding.",
              ],
              [
                "Confidence ≠ Correctness",
                "Model confidence is measured separately from calibration and reliability.",
              ],
              [
                "Benchmark ≠ Deployment",
                "Small controlled datasets are treated as experiments, not evidence of general-world capability.",
              ],
              [
                "Attribution ≠ Causality",
                "Sensitivity diagnostics describe model dependence, not linguistic ground truth.",
              ],
            ].map(([title, body], index) => (
              <div
                key={title}
                className="border-l border-t border-white/10 p-6 sm:min-h-44"
              >
                <div className="mb-8 font-mono text-[10px] text-white/25">
                  0{index + 1}
                </div>
                <h3 className="text-sm font-medium">{title}</h3>
                <p className="mt-3 text-sm leading-6 text-white/40">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="about">
        <div className="mx-auto max-w-7xl px-6 py-16 lg:px-10">
          <div className="flex flex-col justify-between gap-8 border-t border-white/10 pt-8 text-xs text-white/30 sm:flex-row">
            <div>
              LughaLab
              <span className="ml-2 text-white/15">
                African Language &amp; Sign Intelligence
              </span>
            </div>
            <div className="font-mono">TEXT / SPEECH / SIGN / EVALUATION</div>
          </div>
        </div>
      </section>
    </main>
  );
}