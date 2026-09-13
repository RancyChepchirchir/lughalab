# LughaLab

### African Language & Sign Intelligence

LughaLab is a research platform for **low-resource African language AI** across text, speech, and sign. The current work focuses on **Swahili** and **Kenyan Sign Language (KSL)**, with particular attention to representation quality, linguistic sensitivity, domain shift, model adaptation, interpretability, and responsible evaluation.

> **Research question:** How effectively can pretrained multilingual and modality-specific models be adapted to low-resource East African languages, and where do they fail across language, signer, domain, and modality shifts?

---

## Research tracks

| Track | Focus | Current work |
| --- | --- | --- |
| **LughaNLP** | Swahili text | contextual representations, POS tagging, semantic similarity, linguistic stress tests |
| **LughaSpeech** | Swahili speech | zero-shot vs adapted ASR, WER/CER, latency and error decomposition |
| **LughaSign** | Kenyan Sign Language | landmark extraction, Transformer reconstruction, preprocessing audits, reliability and ablation |

```text
                              LughaLab
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
         LughaNLP           LughaSpeech          LughaSign
             │                   │                   │
        Swahili text        Swahili speech      Kenyan Sign
             │                   │               Language
             ▼                   ▼                   ▼
     Representation &       ASR adaptation       Video → landmarks
     linguistic analysis    & error analysis     → recognition
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 ▼
                   Robust African Language AI
```

---

## 1. LughaNLP — Swahili language models

The text track investigates how pretrained African-language encoders represent Swahili and how stable those representations remain under controlled linguistic changes.

**Models used:** SwahBERT, AfroXLM-R, and a Masakhane Swahili POS model.

Current experiments cover contextual embeddings, POS tagging, model comparison, semantic similarity, semantic separation, negation, tense, subject agreement, argument structure, quantification, code-switching, and informal language.

For sentence representations \(z_1\) and \(z_2\), cosine similarity is

\[
\operatorname{sim}(x_1,x_2)
=
\frac{z_1^\top z_2}
{\|z_1\|_2\|z_2\|_2}.
\]

A controlled perturbation is measured by

\[
\Delta = 1-\cos(f_\theta(x),f_\theta(x')).
\]

LughaLab also explores a provisional **Linguistic Sensitivity–Stability score**

\[
LSS =
\mathbb{E}[\Delta\mid\text{meaning-changing}]
-
\mathbb{E}[\Delta\mid\text{meaning-preserving}].
\]

The LSS is an exploratory internal diagnostic, not a validated benchmark.

### Swahili semantic benchmark

The curated benchmark contains paraphrase, topical similarity, unrelated, contrast, informal-language, and code-switching examples. It is deliberately treated as an exploratory diagnostic rather than a gold-standard linguistic dataset.

A simple model-level semantic-separation statistic is

\[
D_m =
\mathbb{E}[s_m\mid\text{similar}]
-
\mathbb{E}[s_m\mid\text{unrelated}].
\]

The broader research question is whether lexical similarity can sometimes masquerade as semantic understanding in low-resource representations.

---

## 2. LughaSpeech — Swahili speech recognition

The speech track compares zero-shot and Swahili-adapted automatic speech recognition.

**Models used:** `openai/whisper-small` and `Finiflowlabs/sauti-asr-v1`.

Evaluation includes standard and informal speech, code-switching, numbers, Word Error Rate (WER), Character Error Rate (CER), Real-Time Factor (RTF), and substitution/deletion/insertion decomposition.

\[
WER=\frac{S+D+I}{N},
\qquad
CER=\frac{S_c+D_c+I_c}{N_c},
\qquad
RTF=\frac{t_{\mathrm{inference}}}{t_{\mathrm{audio}}}.
\]

Adaptation gain is measured as

\[
G_i^{WER}
=
WER_i^{\text{zero-shot}}
-
WER_i^{\text{adapted}}.
\]

LughaLab distinguishes **mean utterance WER** from **corpus-level WER**, and treats orthographic disagreement separately from semantic failure.

### ASR error decomposition

Recognition errors are decomposed into substitution, deletion, and insertion rates:

\[
R_S=\frac{S}{N},
\qquad
R_D=\frac{D}{N},
\qquad
R_I=\frac{I}{N}.
\]

This makes it possible to study not only *how often* an ASR model fails, but *how* it fails.

---

## 3. LughaSign — Kenyan Sign Language

The sign track currently studies landmark-based KSL recognition, preprocessing reconstruction, closed-set reliability, and representation dependence.

The current experimental baseline reconstructs the public `luciayen/afrisign-exp1-ksl-baseline` Landmark Transformer. Its vocabulary is limited to four glosses:

`father` · `hello` · `is` · `my`

It is therefore a **four-class research baseline**, not a general KSL interpreter or translation system.

### Video → landmark pipeline

The native LughaLab pipeline extracts 33 pose landmarks and 21 landmarks from each hand using MediaPipe Holistic. With \((x,y,z)\) coordinates, each frame contains

\[
33(3)+21(3)+21(3)=225
\]

features, giving

\[
X_{\mathrm{native}}\in\mathbb{R}^{T\times225}.
\]

The current implementation pins `mediapipe==0.10.14` for the legacy Holistic API. Migration to MediaPipe Tasks remains future engineering work.

### Native vs checkpoint representation

These representations have the same dimensionality but different semantics:

```text
LughaLab native
┌──────────────────┬──────────────┬──────────────┐
│ Pose: 99         │ Hand: 63     │ Hand: 63     │
└──────────────────┴──────────────┴──────────────┘
                         225

Legacy KSL checkpoint
┌─────────────────────────────────┬──────────────────────┐
│ Hand landmarks: 126             │ Zero padding: 99     │
└─────────────────────────────────┴──────────────────────┘
                         225
```

Hence

\[
\text{shape compatibility}
\neq
\text{representation compatibility}.
\]

The native shoulder-normalized representation is **not** silently fed into the legacy checkpoint.

---

## KSL dataset and checkpoint audit

The original Kaggle landmark source stores variable-length hand-only sequences

\[
X_{\mathrm{raw}}\in\mathbb{R}^{T\times126}.
\]

| Split | Samples | Frame range | Mean frames | Median |
| --- | ---: | ---: | ---: | ---: |
| Train | 572 | 16–43 | 32.038 | 29 |
| Test | 124 | 22–43 | 32.347 | 29 |

The clean test split contains 30 `father`, 30 `hello`, 32 `is`, and 32 `my` samples. The duplicate-containing `dataset3/` source is excluded from the clean evaluation path.

The checkpoint expects

\[
X_{\mathrm{model}}\in\mathbb{R}^{64\times225}.
\]

Its stored normalization vectors have shape `(225,)`. For all 99 padded dimensions,

\[
\mu_j=0,\qquad \sigma_j=1,
\]

so those dimensions remain zero after z-score normalization.

### Transformer reconstruction

Checkpoint inspection reconstructs the following parameter topology:

| Component | Configuration |
| --- | --- |
| Input projection | 225 → 256 |
| CLS token | \(1\times1\times256\) |
| Positional embedding | \(65\times256\) |
| Encoder layers | 4 |
| Attention heads | 8 |
| Feed-forward width | 1024 |
| Classifier | 256 → 4 |

The 65 positions naturally correspond to the CLS token plus 64 temporal tokens:

\[
[\mathrm{CLS},x_1,\ldots,x_{64}].
\]

The reconstructed PyTorch model accepts the checkpoint with `strict=True`. This establishes parameter-topology compatibility, but does **not** by itself prove every non-parameter forward-pass choice is identical.

---

## KSL temporal preprocessing reconstruction

Because the source data are variable length and the checkpoint expects 64 frames, LughaLab tested several plausible temporal transforms over the complete 124-sample clean test split.

| Temporal method | Accuracy | Mean max probability |
| --- | ---: | ---: |
| Repeat-last-frame padding | **99.19%** | 0.5561 |
| Linear interpolation | **99.19%** | 0.5517 |
| Floor resampling | **99.19%** | 0.5515 |
| Nearest-neighbour resampling | **99.19%** | 0.5513 |
| Zero-frame padding | 33.06% | 0.5367 |

Four non-zero strategies reproduce \(123/124=99.19\%\) accuracy. Zero-frame padding falls to \(41/124=33.06\%\).

This strongly rejects naive zero-frame padding, but the tied accuracy does **not** uniquely identify the original temporal transform. LughaLab therefore distinguishes documented architecture, observed dataset structure, checkpoint-derived evidence, and reconstructed preprocessing.

---

## Closed-set reliability

The KSL classifier operates over

\[
\mathcal{Y}=\{\text{father},\text{hello},\text{is},\text{my}\}.
\]

Its softmax necessarily satisfies

\[
\sum_{c\in\mathcal{Y}}P(c\mid X)=1
\]

even for an input unrelated to all four classes. High softmax confidence therefore does not establish that an input is in-distribution.

LughaLab examines maximum probability, prediction margin, and normalized entropy:

\[
p_{\max}=\max_c p(c\mid X),
\]

\[
M=p_{(1)}-p_{(2)},
\]

\[
H_N=
\frac{-\sum_c p_c\log p_c}{\log C}.
\]

Temporal reversal, frame shuffling, static freezing, coordinate perturbation, and an external unlabelled gesture clip are used as **distribution-shift probes**, not labelled as unknown KSL lexical classes. Exploratory ID quantiles are diagnostic thresholds rather than a validated open-set recognition system.

---

## KSL representation ablation

The current ablation study asks which parts of the landmark representation support model decisions. Until source preprocessing independently confirms anatomical ordering, the two 63-dimensional blocks are conservatively denoted

\[
B_1=X_{:,0:63},
\qquad
B_2=X_{:,63:126}.
\]

Ablations include block removal, block isolation, first/second temporal-half removal, and temporal-mean freezing. Removed features are replaced by checkpoint training means so that their standardized values become approximately zero.

Model dependence is measured by the change in probability assigned to the true class:

\[
\Delta p_y
=
p_\theta(y\mid X)
-
p_\theta(y\mid A(X)).
\]

This reveals behavioural changes that top-1 accuracy alone can hide. Ablation effects are interpreted as **model dependence**, not causal linguistic importance.

---

## Project structure

The repository is deliberately separated into the **API/model layer**, **research experiments**, **small reproducible data assets**, and **research documentation**.

```text
lughalab/
├── backend/
│   ├── app/
│   │   ├── api/                         # FastAPI route modules
│   │   │   ├── nlp.py
│   │   │   ├── speech.py
│   │   │   └── sign.py
│   │   ├── models/                      # Pydantic + PyTorch model definitions
│   │   │   ├── benchmark.py
│   │   │   ├── ksl_transformer.py
│   │   │   ├── nlp.py
│   │   │   ├── sign.py
│   │   │   └── speech.py
│   │   ├── services/                    # Inference and preprocessing logic
│   │   │   ├── ksl_checkpoint_adapter.py
│   │   │   ├── ksl_classifier.py
│   │   │   ├── ksl_landmarks.py
│   │   │   ├── ksl_preprocessing.py
│   │   │   ├── model_benchmark.py
│   │   │   ├── swahili_asr.py
│   │   │   ├── swahili_encoder.py
│   │   │   └── swahili_pos.py
│   │   └── main.py                      # FastAPI application
│   ├── tests/
│   └── requirements.txt
│
├── data/
│   ├── ksl/                             # Local KSL research inputs/derived data
│   ├── speech/                          # Local speech benchmark material
│   ├── swahili_linguistic_stress_test.json
│   └── swahili_similarity_benchmark.json
│
├── experiments/
│   ├── results/                         # JSON/CSV/figure research outputs
│   ├── analyse_asr_errors.py
│   ├── analyse_linguistic_sensitivity.py
│   ├── analyse_semantic_separation.py
│   ├── analyse_similarity_benchmark.py
│   ├── audit_ksl_checkpoint.py
│   ├── audit_ksl_checkpoint_statistics.py
│   ├── audit_ksl_dataset_shapes.py
│   ├── audit_ksl_forward_architecture.py
│   ├── plot_linguistic_stress_test.py
│   ├── plot_similarity_benchmark.py
│   ├── reconstruct_ksl_temporal_preprocessing.py
│   ├── run_asr_benchmark.py
│   ├── run_asr_model_comparison.py
│   ├── run_ksl_preprocessing.py
│   ├── run_ksl_representation_ablation.py
│   ├── run_ksl_unknown_rejection.py
│   ├── run_linguistic_stress_test.py
│   └── run_similarity_benchmark.py
│
├── docs/
│   └── research_notes.md
│
├── .gitignore
└── README.md
```

Raw external datasets, downloaded model weights, videos, audio, virtual environments, and caches are intentionally excluded from version control.

---

## API

Start the backend from the repository root:

```bash
cd backend
uvicorn app.main:app --reload
```

The development API runs at `http://127.0.0.1:8000`, with interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

Current research endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/nlp/analyse` | Swahili contextual representation |
| `POST` | `/nlp/pos` | Swahili POS tagging |
| `POST` | `/nlp/benchmark` | representation-model comparison |
| `POST` | `/speech/transcribe` | Swahili ASR |
| `POST` | `/sign/landmarks` | video-to-landmark extraction |

---

## Installation

```bash
git clone https://github.com/RancyChepchirchir/lughalab.git
cd lughalab

python3 -m venv .venv
source .venv/bin/activate

pip install -r backend/requirements.txt
```

The KSL preprocessing pipeline currently pins `mediapipe==0.10.14`.

---

## Data, provenance and ethics

LughaLab does not redistribute large third-party datasets, downloaded checkpoints, raw speech recordings, or external video assets through this repository. External resources should be obtained according to their original licenses and terms.

The project treats speaker/signer diversity, dialect and accent variation, code-switching, domain shift, class imbalance, signer-independent evaluation, provenance, licensing, privacy, and culturally appropriate evaluation as methodological concerns rather than afterthoughts.

In particular, isolated-sign recognition accuracy should not be equated with sign-language competence. A four-class landmark classifier is not a continuous KSL understanding system.

---

## Research principles

1. **Low-resource performance must be measured rather than assumed.**
2. **Model-card metrics are not automatically comparable across datasets.**
3. **Confidence is not calibrated correctness.**
4. **Semantic similarity is not proof of semantic understanding.**
5. **Shape compatibility is not representation compatibility.**
6. **Closed-set recognition is not open-set recognition.**
7. **Signer-independent generalisation matters for sign-language evaluation.**
8. **Code-switching, informal language, dialect and domain shift matter.**
9. **Data provenance, licensing, privacy and community representation matter.**
10. **Undocumented preprocessing should be reconstructed cautiously rather than silently assumed.**
11. **Ablation measures model dependence, not automatically linguistic causality.**
12. **Strong accuracy on a tiny vocabulary must not be generalized to an entire language.**

---

## Current status

The first research cycle now includes Swahili representation analysis and POS inference, cross-model semantic comparison, controlled linguistic stress testing, zero-shot/adapted Swahili ASR, ASR error decomposition, KSL landmark extraction, checkpoint reconstruction, source-data auditing, temporal-preprocessing reconstruction, closed-set reliability diagnostics, distribution-shift probes, and representation ablation.

Next research stages include temporal/landmark attribution, signer-independent KSL evaluation, larger-vocabulary modelling, richer Swahili robustness benchmarks, and eventually continuous sign and multimodal translation research.

---

## Long-term direction

```text
                    ┌─────────────────────────┐
                    │        LughaLab         │
                    │ African Language & Sign │
                    │      Intelligence       │
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
    ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
    │    TEXT     │       │   SPEECH    │       │    SIGN     │
    │ Swahili NLP │       │ Swahili ASR │       │     KSL     │
    └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
           │                     │                     │
           ▼                     ▼                     ▼
    Representation          Transcription         Sign / Gloss
       & NLP tasks            & speech             modelling
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ Multimodal African      │
                    │ Language Intelligence   │
                    └────────────┬────────────┘
                                 │
                          ┌──────┴──────┐
                          ▼             ▼
                       Swahili       English
```

The long-term goal is not to collect independent demos, but to investigate language intelligence across

\[
\text{text}\leftrightarrow\text{speech}\leftrightarrow\text{sign}.
\]

---

## Disclaimer

LughaLab is a research project. The current KSL classifier is **not a sign-language interpreter** and should not be used for medical, legal, emergency, safeguarding, financial, or other high-stakes communication.

Predictions must be interpreted within the limitations of the training data, vocabulary, signer distribution, preprocessing assumptions, and model architecture.

---

## Author

**Rancy Chepchirchir**

Research interests: low-resource NLP · multimodal learning · sign-language AI · representation learning · trustworthy AI · scientific machine learning
