# LughaLab

**African Language & Sign Intelligence**

LughaLab is a research-oriented platform for studying low-resource African
language technologies across text, speech, and sign language.

The project currently focuses on **Swahili** and **Kenyan Sign Language (KSL)**,
with an emphasis on representation quality, robustness, domain shift,
adaptation, interpretability, and responsible evaluation rather than
demo-only inference.

---

## Research Question

> How effectively can pretrained multilingual and modality-specific foundation
> models be adapted to low-resource East African languages, and where do they
> fail when evaluated across language, signer, domain, and modality shifts?

LughaLab is organised around three complementary research tracks:

```text
                         LUGHALAB
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
      LughaNLP           LughaSpeech         LughaSign
         │                   │                   │
      Swahili              Swahili              KSL
       Text                 Speech          Video/Landmarks
         │                   │                   │
   Representation        Whisper ASR       Landmark Models
   POS / Semantics       Adapted ASR       Sign Recognition
   Stress Testing        Error Analysis    Reliability
```

---

## 1. LughaNLP — Swahili Language Models

The text track investigates representation learning and linguistic sensitivity
in pretrained African-language models.

Current models include:

- SwahBERT 
- AfroXLM-R 
- Masakhane Swahili POS models 

Implemented experiments include:

- contextual Swahili embeddings; 
- Swahili POS tagging; 
- SwahBERT vs AfroXLM-R representation comparison; 
- semantic similarity benchmarking; 
- semantic separation diagnostics; 
- linguistic stress testing; 
- negation; 
- tense; 
- subject agreement; 
- argument structure; 
- quantification; 
- code-switching; 
- informal-language evaluation. 

For two sentences x1​ and x2​, representation similarity is measured as

sim(x1​,x2​)=∥z1​∥2​∥z2​∥2​z1⊤​z2​​. 

Representational change under a linguistic perturbation is measured using

Δ=1−cos(fθ​(x),fθ​(x′)). 

A provisional Linguistic Sensitivity–Stability score is also explored:

LSS=E[Δ∣meaning-changing]−E[Δ∣meaning-preserving]. 

This metric is currently exploratory and is not presented as a validated
benchmark.

---

## 2. LughaSpeech — Swahili Speech Recognition

The speech track studies both zero-shot and adapted automatic speech
recognition for Swahili.

Models currently evaluated include:

- `openai/whisper-small` 
- `Finiflowlabs/sauti-asr-v1` 

Evaluation includes:

- Word Error Rate; 
- Character Error Rate; 
- Real-Time Factor; 
- substitution/deletion/insertion decomposition; 
- standard Swahili; 
- informal speech; 
- code-switching; 
- numbers and structured expressions. 

Word Error Rate is defined as

WER=NS+D+I​, 

where:

- S = substitutions; 
- D = deletions; 
- I = insertions; 
- N = number of reference words. 

The comparison between zero-shot and adapted ASR is measured using

GiWER​=WERizero−shot​−WERiadapted​. 

A central concern of this track is that:

orthographic error=semantic failure. 

---

## 3. LughaSign — Kenyan Sign Language

The sign-language track studies landmark-based Kenyan Sign Language
recognition, representation quality, temporal structure, robustness, and
 closed-set reliability.

The current baseline is reconstructed from the public

```
luciayen/afrisign-exp1-ksl-baseline
```

checkpoint.

The model is a Landmark Transformer operating on sequences of shape

64×225. 

The reconstructed checkpoint architecture contains:

- 225-dimensional landmark input; 
- 256-dimensional embeddings; 
- 8 attention heads; 
- 4 Transformer encoder layers; 
- 1024-dimensional feed-forward layers; 
- learned positional embeddings; 
- final LayerNorm; 
- four output classes. 

The baseline vocabulary is restricted to:

```
father
```

hello

is

my

It must therefore **not** be interpreted as a general Kenyan Sign Language
 translation or interpretation system.

---

## KSL Video-to-Landmark Pipeline

LughaLab includes a MediaPipe-based video processing pipeline that extracts:

- 33 body pose landmarks; 
- 21 left-hand landmarks; 
- 21 right-hand landmarks. 

Each landmark contains

(x,y,z) 

coordinates.

Therefore the native LughaLab representation contains:

33×3+21×3+21×3=225 

features per frame.

The resulting native representation is

X∈RT×225. 

The current pipeline uses MediaPipe Holistic and therefore pins:

```
mediapipe==0.10.14
```

for compatibility with the legacy `mp.solutions.holistic` API.

Migration to the newer MediaPipe Tasks APIs is a future engineering task.

---

## Native and Legacy KSL Representations

LughaLab distinguishes between two different 225-dimensional representations.

### Native LughaLab representation

```
[ pose 99 | hand block 63 | hand block 63 ]
```

This representation is produced directly from MediaPipe Holistic.

### Legacy checkpoint representation

```
[ hand landmarks 126 | zero padding 99 ]
```

The external KSL checkpoint was trained using hand-only features occupying the
 first 126 dimensions.

The remaining 99 dimensions are structural zero padding.

This distinction is important because:

shape compatibility=semantic feature compatibility. 

A tensor having shape

64×225 

does not automatically mean that its features correspond to those expected by
 the pretrained model.

---

# KSL Dataset Audit

The original KSL landmark dataset contains variable-length, hand-only landmark
 sequences:

Xraw​∈RT×126. 

The audited clean training split contains:

```
father : 140
```

hello  : 140

is     : 146

my     : 146

for a total of:

572 

training sequences.

Training sequence lengths range from:

16≤T≤43. 

Observed statistics include:

```
minimum: 16
```

maximum: 43

mean:    32.038

median:  29

The clean test split contains:

```
father : 30
```

hello  : 30

is     : 32

my     : 32

for a total of:

124 

test sequences.

Test sequence lengths range from:

22≤T≤43. 

Observed statistics include:

```
minimum: 22
```

maximum: 43

mean:    32.347

median:  29

The checkpoint, however, expects:

Xmodel​∈R64×225. 

This demonstrates that additional temporal and feature preprocessing is required
 between the source dataset and the Transformer.

---

# KSL Checkpoint Audit

The reconstructed checkpoint contains:

```
epoch
```

val\_acc

model

l2i

mean

std

The label mapping is:

```
father -> 0
```

hello  -> 1

is     -> 2

my     -> 3

The stored feature statistics have shape:

(225,). 

For the first 126 hand features, the checkpoint stores non-trivial training
 statistics.

Observed summary values include:

```
mean absolute mean: 0.1862
```

mean standard deviation: 0.0753

zero std features: 0

For dimensions 126:225:

```
max absolute mean: 0.0
```

max absolute std:  1.0

zero means:        99 / 99

Thus, for every padded feature j,

μj​=0,σj​=1. 

After checkpoint z-score normalization,

zt,j​=σj​xt,j​−μj​​, 

the padded dimensions remain exactly zero.

---

# KSL Temporal Reconstruction

The original source sequences are variable-length, while the model consumes
 exactly 64 frames.

The undocumented transformation can therefore be written abstractly as:

R64​\:RT×126→R64×126. 

LughaLab evaluates several plausible temporal preprocessing strategies rather
 than silently assuming one.

The complete clean 124-sample test split produced the following results:

| Temporal methodAccuracyMean max probability |        |        |
| ------------------------------------------- | ------ | ------ |
| Repeat-last-frame padding                   | 99.19% | 0.5561 |
| Linear interpolation                        | 99.19% | 0.5517 |
| Floor resampling                            | 99.19% | 0.5515 |
| Nearest-neighbour resampling                | 99.19% | 0.5513 |
| Zero-frame padding                          | 33.06% | 0.5367 |

Four distinct temporal transformations therefore reproduce almost identical
 classification performance:

123/124=99.19%. 

Naive zero padding produces only:

41/124=33.06%. 

This strongly rejects zero-frame temporal padding but does **not** uniquely
 identify the original temporal normalization method.

The four-class task is sufficiently separable that several transformations can
 preserve the model's predictions.

LughaLab therefore distinguishes between:

- documented architecture; 
- observed dataset structure; 
- checkpoint-derived evidence; 
- reconstructed preprocessing. 

Exact temporal preprocessing equivalence is not claimed where source evidence
 is insufficient.

---

# Checkpoint Transformer Reconstruction

The checkpoint architecture was reconstructed directly from its state
 dictionary.

Important parameter shapes include:

```
projection:
```

256 x 225

CLS token:

1 x 1 x 256

positional embedding:

65 x 256

Transformer layers:

4

attention embedding:

256

attention heads:

8

feed-forward dimension:

1024

classifier:

4 x 256

The 65 positional embeddings correspond naturally to:

1+64 

tokens:

[CLS,x1​,…,x64​]. 

The reconstructed model successfully loads the checkpoint using:

```
model.load_state_dict(
```

state\_dict,

strict=True,

)

This confirms parameter-topology compatibility.

However:

strict parameter compatibility=proof of identical computation graph. 

Non-parameter implementation choices such as activation functions or
 pre/post-LayerNorm placement are not fully determined by state-dict shapes.

---

# Closed-Set Reliability

The current KSL classifier operates over:

Y={father,hello,is,my}. 

For every input X,

c∈Y∑​P(c∣X)=1. 

This remains true even when an input is unrelated to all four trained classes.

Consequently, the model will always produce one of the four labels.

A high softmax probability therefore does **not** prove that the input belongs
 to the training distribution.

LughaLab investigates three descriptive uncertainty quantities.

### Maximum probability

pmax​(X)=cmax​p(c∣X). 

### Prediction margin

M(X)=p(1)​(X)−p(2)​(X). 

### Normalized entropy

HN​(X)=log∣Y∣−∑c​pc​(X)logpc​(X)​. 

The full clean test split provides the reference distribution for these
 statistics.

Exploratory rejection boundaries are derived from:

Q0.05ID​ 

for maximum probability and prediction margin, and

Q0.95ID​ 

for predictive entropy.

These thresholds are exploratory rather than validated open-set recognition
 criteria.

---

# KSL Distribution-Shift Probes

LughaLab evaluates several synthetic perturbations:

```
temporal reversal
```

temporal frame shuffling

static-frame freezing

coordinate perturbation

An external unlabelled gesture video is also retained as a pipeline
 out-of-distribution control.

These examples are **not** labelled as new or unknown KSL words.

They are stress tests designed to examine whether the closed-set classifier
 recognizes departures from its training distribution.

The distinction is important:

distribution shift probe=unknown lexical class. 

---

# KSL Representation Ablation

LughaLab also investigates which components of the input representation drive
 model decisions.

Because the internal anatomical ordering of the two 63-dimensional hand blocks
 has not yet been independently established from the source preprocessing code,
 the blocks are currently denoted:

B1​=X:,0:63​, 

and

B2​=X:,63:126​. 

They are intentionally not labelled as left or right hand without stronger
 source evidence.

Current ablations include:

```
clean
```

block\_1\_removed

block\_2\_removed

block\_1\_only

block\_2\_only

temporal\_mean\_freeze

first\_half\_only

second\_half\_only

Feature removal uses the checkpoint training means.

Because normalization is:

zj​=σj​xj​−μj​​, 

setting:

xj​=μj​ 

produces:

zj​=0. 

This provides a more principled neutralization method than simply inserting raw
 zeros.

---

## True-Class Probability Ablation

Accuracy alone can hide substantial changes in model behaviour.

For an ablation A, LughaLab therefore also measures:

Δpy​=pθ​(y∣X)−pθ​(y∣A(X)). 

A large positive value means that the ablated component strongly supported the
 correct class.

This enables the analysis to detect model dependence even when top-1
 classification remains unchanged.

---

## Temporal Mean Freezing

Temporal mean freezing replaces the sequence

X=(x1​,…,x64​) 

with its mean landmark configuration:

xˉ=641​t=1∑64​xt​. 

The resulting sequence is:

Xfreeze​=(xˉ,xˉ,…,xˉ). 

This removes most explicit temporal dynamics while approximately retaining
 average hand configuration.

If classification remains strong after this operation, that would suggest
 substantial reliance on static spatial configuration.

Such a conclusion is made only from experimental results and is not assumed in
 advance.

Ablation effects are interpreted as **model dependence**, not causal linguistic
 importance.

---

# Swahili Semantic Benchmark

LughaLab includes a small curated Swahili similarity benchmark containing
 examples from categories such as:

```
paraphrase
```

topical similarity

unrelated

contrast

informal language

code-switching

It is used as an exploratory diagnostic rather than as a gold-standard
 linguistic benchmark.

For model m, a simple semantic-separation quantity is:

Dm​=E[sm​∣similar]−E[sm​∣unrelated]. 

Additional analysis compares semantically similar sentences against contrast
 and unrelated cases.

The experiments are designed partly to investigate whether lexical overlap can
 masquerade as semantic understanding.

---

# Swahili Linguistic Stress Test

The controlled linguistic stress test currently evaluates:

```
negation
```

tense

subject agreement

argument structure

quantification

code-switching

informal language

The representational change induced by a perturbation is:

Δk​=1−cos(fθ​(x),fθ​(x′)). 

Meaning-changing perturbations ideally produce stronger representational
 changes than meaning-preserving variants.

The current Linguistic Sensitivity–Stability score is:

LSSm​=Sm​−Im​, 

where

Sm​=E[Δ∣meaning-changing] 

and

Im​=E[Δ∣meaning-preserving]. 

This is an exploratory internal metric and is not currently claimed as a novel
 validated benchmark.

---

# Swahili Speech Benchmark

LughaLab includes a small controlled ASR benchmark with categories including:

```
standard speech
```

informal speech

code-switching

numbers

The current evaluation compares a zero-shot Whisper baseline against a
 Swahili-adapted model.

Metrics include:

WER=NS+D+I​ 

and

CER=Nc​Sc​+Dc​+Ic​​. 

Runtime efficiency is summarized using:

RTF=taudio​tinference​​. 

The project distinguishes between:

```
mean per-utterance WER
```

and

```
corpus-level WER
```

because these are not numerically equivalent.

---

# ASR Error Decomposition

Speech-recognition errors are decomposed into:

- substitutions; 
- deletions; 
- insertions. 

Corresponding rates are:

RS​=NS​,RD​=ND​,RI​=NI​. 

This allows LughaLab to investigate not only how often an ASR model fails, but
 also **how** it fails.

---

# Project Structure

```
lughalab/
```

│

├── backend/

│   ├── app/

│   │   ├── api/

│   │   │   ├── nlp.py

│   │   │   ├── speech.py

│   │   │   └── sign.py

│   │   │

│   │   ├── models/

│   │   │   ├── nlp.py

│   │   │   ├── speech.py

│   │   │   ├── sign.py

│   │   │   ├── benchmark.py

│   │   │   └── ksl\_transformer.py

│   │   │

│   │   ├── services/

│   │   │   ├── swahili\_encoder.py

│   │   │   ├── swahili\_pos.py

│   │   │   ├── model\_benchmark.py

│   │   │   ├── swahili\_asr.py

│   │   │   ├── ksl\_landmarks.py

│   │   │   ├── ksl\_preprocessing.py

│   │   │   ├── ksl\_checkpoint\_adapter.py

│   │   │   └── ksl\_classifier.py

│   │   │

│   │   └── main.py

│   │

│   ├── tests/

│   └── requirements.txt

│

├── data/

│   ├── swahili\_similarity\_benchmark.json

│   ├── swahili\_linguistic\_stress\_test.json

│   ├── speech/

│   └── ksl/

│

├── experiments/

│   ├── results/

│   │

│   ├── run\_similarity\_benchmark.py

│   ├── analyse\_similarity\_benchmark.py

│   ├── plot\_similarity\_benchmark.py

│   ├── analyse\_semantic\_separation.py

│   │

│   ├── run\_linguistic\_stress\_test.py

│   ├── plot\_linguistic\_stress\_test.py

│   ├── analyse\_linguistic\_sensitivity.py

│   │

│   ├── run\_asr\_benchmark.py

│   ├── run\_asr\_model\_comparison.py

│   ├── analyse\_asr\_errors.py

│   │

│   ├── run\_ksl\_preprocessing.py

│   ├── audit\_ksl\_checkpoint.py

│   ├── audit\_ksl\_dataset\_shapes.py

│   ├── audit\_ksl\_checkpoint\_statistics.py

│   ├── reconstruct\_ksl\_temporal\_preprocessing.py

│   ├── audit\_ksl\_forward\_architecture.py

│   ├── run\_ksl\_unknown\_rejection.py

│   └── run\_ksl\_representation\_ablation.py

│

├── docs/

│   └── research\_notes.md

│

├── README.md

└── .gitignore

---

# API

Start the FastAPI backend with:

```
cd backend
```

uvicorn app.main\:app --reload

The API is then available at:

```
http://127.0.0.1:8000
```

Interactive OpenAPI documentation is available at:

```
http://127.0.0.1:8000/docs
```

Current research endpoints include:

```
POST /nlp/analyse
```

POST /nlp/pos

POST /nlp/benchmark

POST /speech/transcribe

POST /sign/landmarks

---

# Installation

Clone the repository:

```
git clone https://github.com/RancyChepchirchir/lughalab.git
```

cd lughalab

Create a virtual environment:

```
python3 -m venv .venv
```

source .venv/bin/activate

Install dependencies:

```
pip install -r backend/requirements.txt
```

The current KSL preprocessing pipeline intentionally pins:

```
mediapipe==0.10.14
```

because it relies on the legacy MediaPipe Holistic API.

---

# Data and Model Provenance

LughaLab does not redistribute large third-party datasets, model checkpoints,
 raw speech recordings, or downloaded video assets through the repository.

External resources are obtained independently according to their original
 licenses and terms.

Small derived research outputs may be retained when appropriate for
 reproducibility.

Dataset and model provenance are treated as part of the research methodology
 rather than as an implementation detail.

---

# Data Governance and Ethics

Low-resource language technology can easily reproduce gaps in representation.

LughaLab therefore treats the following as core evaluation concerns:

- speaker and signer diversity; 
- dialect variation; 
- accent variation; 
- code-switching; 
- domain shift; 
- class imbalance; 
- signer-independent evaluation; 
- data provenance; 
- licensing; 
- privacy; 
- culturally appropriate interpretation; 
- native-speaker and native-signer evaluation. 

For sign-language systems in particular, recognition performance should not be
 equated with linguistic competence.

A model that recognizes a small vocabulary of isolated signs is not equivalent
 to a system capable of continuous sign-language understanding.

---

# Research Principles

LughaLab follows several principles throughout the project.

1. **Low-resource performance must be measured rather than assumed.** 
2. **Model-card metrics are not automatically comparable across datasets.** 
3. **Confidence is not equivalent to calibrated correctness.** 
4. **Semantic similarity is not proof of semantic understanding.** 
5. **Shape compatibility is not equivalent to representation compatibility.** 
6. **Closed-set recognition is not open-set recognition.** 
7. **Sign-language evaluation should consider signer-independent**
    **generalisation.** 
8. **Code-switching, informal language, dialect, and domain shift matter.** 
9. **Data provenance, licensing, privacy, and community representation matter.** 
10. **Undocumented preprocessing should be reconstructed cautiously rather than**
     **silently assumed.** 
11. **Ablation describes model dependence, not automatically linguistic**
     **causality.** 
12. **Strong accuracy on a small vocabulary should not be generalized to a**
     **complete language.** 

---

# Current Status

LughaLab is under active research development.

Completed first-cycle investigations include:

- Swahili contextual representation analysis; 
- Swahili POS inference; 
- SwahBERT vs AfroXLM-R comparison; 
- curated semantic similarity benchmarking; 
- semantic separation diagnostics; 
- controlled linguistic stress testing; 
- sensitivity-versus-invariance analysis; 
- Swahili zero-shot ASR; 
- adapted Swahili ASR comparison; 
- WER/CER/RTF analysis; 
- ASR substitution/deletion/insertion decomposition; 
- KSL video-to-landmark extraction; 
- KSL landmark preprocessing; 
- KSL checkpoint auditing; 
- Transformer architecture reconstruction; 
- source KSL dataset auditing; 
- temporal preprocessing reconstruction; 
- closed-set reliability diagnostics; 
- KSL distribution-shift testing; 
- KSL representation ablation. 

Future work includes:

- temporal attribution; 
- landmark-level attribution; 
- signer-independent evaluation; 
- larger KSL vocabularies; 
- continuous sign modelling; 
- sign-to-gloss pipelines; 
- KSL-to-Swahili and KSL-to-English translation research; 
- richer Swahili semantic benchmarks; 
- dialect and accent robustness; 
- frontend research visualisation; 
- full LughaLab technical documentation. 

---

# Long-Term Research Direction

The intended multimodal architecture is:

```
                         AFRICAN LANGUAGE AI
```

                                 │

             ┌───────────────────┼───────────────────┐

             │                   │                   │

            TEXT               SPEECH               SIGN

             │                   │                   │

      SwahBERT / XLM-R      Whisper / Wav2Vec   Video / Landmarks

             │                   │                   │

       NLP Tasks             Swahili ASR       Landmark Encoder

             │                   │                   │

             └──────────┬────────┘                   │

                        │                            ▼

                        │                         KSL Gloss

                        │                            │

                        └──────────────┬─────────────┘

                                       │

                              ┌────────┴────────┐

                              ▼                 ▼

                           Swahili            English

The eventual goal is not simply to combine independent demos, but to study how
 language representations interact across:

text↔speech↔sign. 

---

# Disclaimer

LughaLab is a research project.

The current KSL classifier is not a sign-language interpreter.

It should not be used for:

- medical communication; 
- legal interpretation; 
- emergency communication; 
- safeguarding decisions; 
- financial decisions; 
- other high-stakes communication. 

Predictions should always be interpreted within the limitations of the
 training data, vocabulary, signer distribution, preprocessing assumptions, and
model architecture.

---

# Repository

GitHub:

```
https://github.com/RancyChepchirchir/lughalab
```

---

# Author

**Rancy Chepchirchir**

Research interests include:

- artificial intelligence; 
- low-resource NLP; 
- multimodal learning; 
- sign-language AI; 
- scientific machine learning; 
- trustworthy AI; 
- responsible AI; 
- representation learning.
