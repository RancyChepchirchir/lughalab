# LughaLab Research Notes

## Project identity

**LughaLab — African Language & Sign Intelligence**

A multimodal research environment for low-resource African language technology,
with an initial focus on Swahili and Kenyan Sign Language.

---

## Research direction

LughaLab is not intended to be a collection of unrelated model demos.

The central research question is:

> How effectively can pretrained multilingual and modality-specific models be
> adapted to low-resource East African languages, and where do they fail under
> changes in domain, language variety, speaker, signer and modality?

The project will span:

- Swahili NLP
- Swahili speech
- Kenyan Sign Language
- multimodal translation
- model comparison
- low-resource adaptation
- out-of-domain evaluation
- signer-independent evaluation
- data provenance and model limitations

---

## Current architecture

### Representation model

SwahBERT is used as a Swahili-specific contextual encoder.

For contextual representations

\[
H = (h_1, h_2, \ldots, h_n),
\]

a masked mean sentence representation is computed as

\[
z =
\frac{\sum_i m_i h_i}
{\sum_i m_i}.
\]

This representation is not assumed to be a calibrated semantic embedding.

### POS tagging

A task-specific AfroXLM-R Swahili POS model is used for linguistic tagging.

Subword predictions are collapsed to one prediction per original word.

Model softmax confidence is not interpreted as calibrated probability of
linguistic correctness.

### Similarity

For two sentence representations \(z_1\) and \(z_2\),

\[
\operatorname{sim}(x_1,x_2)
=
\frac{z_1^\top z_2}
{\|z_1\|_2 \|z_2\|_2}.
\]

Current similarity experiments are exploratory diagnostics only.

The curated starter set contains:

- paraphrases
- topical similarity
- unrelated text
- contrast
- informal Swahili
- Swahili-English code switching

It is not currently a gold-standard semantic textual similarity benchmark.

---

## Important scientific cautions

1. Higher cosine similarity does not necessarily imply a better semantic model.
2. Mean-pooled pretrained encoders are not equivalent to sentence-transformer models.
3. High model confidence does not imply calibrated correctness.
4. Performance on standard Swahili should not be assumed to generalise to informal,
   dialectal or code-switched language.
5. KSL random-split performance must not be confused with signer-independent generalisation.
6. Dataset provenance and native-language evaluation will be treated as first-class concerns.

---

## Planned modalities

### Text

Swahili representation learning, POS, NER, sentiment, semantic similarity,
question answering and translation.

### Speech

Swahili ASR, accent and dialect evaluation, and speech-to-NLP pipelines.

### Sign

Kenyan Sign Language video, landmark extraction, isolated-sign recognition,
sign-to-gloss modelling and KSL-to-English/Swahili translation.

---

## Future evaluation

Candidate evaluations include:

\[
\Delta_{\mathrm{sim}}
=
\mathrm{sim}_{A}
-
\mathrm{sim}_{B}
\]

and latency difference

\[
\Delta_t
=
t_A - t_B.
\]

Future experiments should additionally report task-appropriate metrics such as:

- accuracy
- macro-F1
- weighted-F1
- precision
- recall
- calibration error
- signer-independent accuracy
- BLEU / chrF where appropriate
- WER / CER for speech
- domain-shift degradation

---

## Long-term system

LughaLab should evolve into a research workstation where users can inspect:

- model predictions
- uncertainty
- tokenisation
- embeddings
- model agreement
- model disagreement
- domain shift
- language variety
- code switching
- speech output
- KSL predictions
- cross-modal translation

---

## Linguistic Sensitivity vs Invariance

The Swahili linguistic stress test introduces controlled transformations to
sentence pairs.

For encoder \(m\), define the representational shift

\[
\Delta_m(x,x')
=
1 -
\cos(
f_m(x),
f_m(x')
).
\]

Meaning-changing transformations should ideally produce noticeable movement in
representation space, while meaning-preserving transformations should ideally
produce relatively little movement.

We therefore distinguish

\[
S_m
=
\mathbb{E}
[
\Delta_m
\mid
\text{meaning-changing}
]
\]

and

\[
I_m
=
\mathbb{E}
[
\Delta_m
\mid
\text{meaning-preserving}
].
\]

A provisional diagnostic quantity is

\[
LSS_m
=
S_m-I_m.
\]

This is referred to provisionally as the Linguistic Sensitivity-Stability
score.

It is not currently claimed as a validated or novel metric.

Validation would require substantially larger datasets, native-speaker review,
statistical uncertainty estimates, comparison with established semantic and
linguistic probing methods, and literature review.

The more important outputs at the present stage are individual failure cases:

1. meaning-changing transformations with unusually small representational
   shifts;
2. meaning-preserving transformations with unusually large representational
   shifts.

These cases expose the distinction between lexical similarity, compositional
semantics and robustness to surface-form variation.

---

## Swahili ASR Error Decomposition

ASR quality is evaluated not only through aggregate word error rate but also
through the composition of recognition errors.

For a reference transcript containing \(N\) words,

\[
WER
=
\frac{S+D+I}{N},
\]

where \(S\), \(D\), and \(I\) denote substitutions, deletions, and insertions.

The corresponding diagnostic rates are

\[
R_S=\frac{S}{N},
\qquad
R_D=\frac{D}{N},
\qquad
R_I=\frac{I}{N}.
\]

This decomposition distinguishes systems that hallucinate additional content,
omit spoken content, or replace spoken words with alternative lexical forms.

This distinction is particularly important for multilingual and code-switched
speech. A word-level substitution can reflect several qualitatively different
phenomena, including acoustic confusion, orthographic variation, borrowing,
phonological adaptation, or genuine semantic recognition failure.

LughaLab therefore preserves raw reference and hypothesis transcripts alongside
normalised forms and quantitative error metrics.

The current benchmark is intentionally small and exploratory. Error frequencies
must not be interpreted as representative distributions of Swahili speakers,
dialects, accents, or recording conditions.

---

## Kenyan Sign Language — Landmark Representation

The third LughaLab modality investigates Kenyan Sign Language.

The initial pipeline deliberately separates visual preprocessing from linguistic
recognition:

\[
\text{video}
\rightarrow
\text{landmarks}
\rightarrow
\text{temporal representation}
\rightarrow
\text{recognition}.
\]

Each video frame is represented using 33 body-pose landmarks, 21 left-hand
landmarks and 21 right-hand landmarks.

Each landmark contains three spatial coordinates, producing

\[
d
=
(33+21+21)\times3
=
225
\]

features per frame.

A video containing \(T\) frames is therefore represented as

\[
X\in\mathbb{R}^{T\times225}.
\]

Missing hand detections are initially represented by zero-valued coordinates
rather than removing the corresponding frames. This preserves temporal
alignment but introduces an explicit missing-observation pattern that may
affect downstream recognition.

Landmark detection quality is measured using

\[
R_{\mathrm{detect}}
=
\frac{N_{\mathrm{frames\ with\ detected\ hands}}}
{N_{\mathrm{frames}}}.
\]

This distinction is methodologically important because downstream recognition
failure can originate either from the recognition model or from upstream
landmark extraction.

LughaLab initially studies isolated-sign recognition rather than claiming
general KSL translation capability.

For isolated recognition,

\[
f_\theta:
\mathbb{R}^{T\times225}
\rightarrow
\{1,\ldots,C\}.
\]

Continuous sign-language translation instead requires mapping a variable-length
visual sequence to a variable-length linguistic sequence and introduces
substantially different modelling and evaluation problems.

The current landmark pipeline is therefore treated as preprocessing and must
not itself be described as KSL understanding or translation.

---

## KSL Temporal and Geometric Preprocessing

Raw sign videos contain variable numbers of frames. Landmark sequences therefore
initially have shape

\[
X\in\mathbb{R}^{T\times225}.
\]

For fixed-length transformer inference, LughaLab initially maps each sequence to

\[
X'\in\mathbb{R}^{64\times225}.
\]

Each frame is approximately centred using the midpoint of the left and right
shoulders,

\[
c_t
=
\frac{p_t^{(L)}+p_t^{(R)}}{2},
\]

and scaled using shoulder width,

\[
s_t
=
\left\|
p_{t,xy}^{(L)}
-
p_{t,xy}^{(R)}
\right\|_2.
\]

Valid landmarks are transformed according to

\[
\tilde p_{t,j}
=
\frac{p_{t,j}-c_t}{s_t}.
\]

Missing landmarks remain explicitly zero-valued after geometric normalization
so that missing observations are not transformed into artificial coordinates.

Temporal linear interpolation provides the initial fixed-length baseline.

This creates an important methodological issue: zero-valued missing landmarks
can be treated by ordinary interpolation as genuine spatial coordinates.
Future experiments should therefore compare zero filling, mask-aware temporal
interpolation and short-gap interpolation.

Preprocessing quality is considered part of the sign-recognition system rather
than an invisible implementation detail.

---

## KSL Checkpoint Compatibility Audit

The first public KSL recognition baseline evaluated by LughaLab is
`luciayen/afrisign-exp1-ksl-baseline`.

The model is a custom LandmarkTransformer with an expected input shape

\[
X\in\mathbb{R}^{64\times225}.
\]

However, shape compatibility does not imply semantic feature compatibility.

The original KSL dataset contains hands-only landmarks:

\[
42\times3=126
\]

features per frame. These occupy dimensions \(0{:}126\), while dimensions
\(126{:}225\) are zero padding.

This differs from the native LughaLab Holistic representation,

\[
X_t=
[
\text{pose}_{99},
\text{left-hand}_{63},
\text{right-hand}_{63}
],
\]

which also has 225 dimensions.

Consequently, the native LughaLab representation must not be passed directly
to the checkpoint merely because the tensor shapes agree.

A compatibility adapter therefore constructs

\[
X_t^{(\mathrm{legacy})}
=
[
\text{left-hand}_{63},
\text{right-hand}_{63},
\mathbf 0_{99}
].
\]

The checkpoint additionally stores feature-wise training statistics
\(\mu_j\) and \(\sigma_j\). Inference features are standardized using

\[
z_{t,j}
=
\frac{x_{t,j}-\mu_j}{\sigma_j}.
\]

Zero-variance dimensions require protected division.

The native LughaLab representation and the legacy checkpoint representation
are retained separately. This prevents model-specific preprocessing from
silently becoming the canonical KSL representation of the wider project.

The public baseline contains only four glosses:

- father
- hello
- is
- my

Its reported 100% validation result concerns this small four-class task and
must not be presented as evidence of general Kenyan Sign Language recognition.

The current compatibility adapter uses linear temporal resampling to obtain
64 frames. This establishes dimensional compatibility but is not claimed to
exactly reproduce the original dataset preprocessing unless that procedure
is independently verified.

---

## KSL Closed-Set Reliability Diagnostic

The first recognition experiment evaluates not only classification accuracy but
also the behaviour of the four-class KSL baseline outside its known label
space.

The model implements a closed-set classifier over

\[
\mathcal{Y}
=
\{
\text{father},
\text{hello},
\text{is},
\text{my}
\}.
\]

For every input \(X\),

\[
\sum_{c\in\mathcal{Y}}
p(c\mid X)=1,
\]

even if the input does not correspond to any supported KSL gloss.

This means the predicted class alone cannot establish whether the observation
belongs to the model's training domain.

LughaLab therefore records maximum softmax probability,

\[
p_{\max}
=
\max_c p(c\mid X),
\]

the margin between the two highest probabilities,

\[
M
=
p_{(1)}-p_{(2)},
\]

and normalized predictive entropy,

\[
H_N(p)
=
\frac{
-\sum_c p_c\log p_c
}{
\log C
}.
\]

A small diagnostic compares genuine test examples from each of the four known
classes with an external video used solely as an out-of-distribution negative
control.

The external video is not labelled as Kenyan Sign Language and must not be
described as an example of whichever class the closed-set model predicts.

High confidence on the OOD control would demonstrate that softmax confidence
alone is insufficient for unknown-sign rejection.

Conversely, lower confidence or higher entropy on the OOD sample would be
suggestive but would not validate an OOD detector.

This experiment contains only four in-distribution examples and one external
control. It is therefore an exploratory reliability diagnostic rather than a
formal OOD benchmark.

## KSL Source Dataset and Temporal Reconstruction

An audit of the original Kaggle KSL landmark corpus established that source
samples are variable-length hand-only sequences rather than fixed model-ready
tensors.

Across the clean training split, sequence lengths range from 16 to 43 frames,
with a median length of 29 frames. The clean test split ranges from 22 to 43
frames, again with 126 features per frame.

Thus,

\[
X_{\mathrm{raw}}\in\mathbb{R}^{T\times126},
\]

while the trained LandmarkTransformer consumes

\[
X_{\mathrm{model}}\in\mathbb{R}^{64\times225}.
\]

Checkpoint statistics independently confirm that dimensions \(126:225\) are
structural padding: their training mean is zero and their stored standard
deviation is one. Consequently, these 99 dimensions remain zero after
normalisation.

The remaining undocumented preprocessing operation is therefore temporal
normalisation from variable \(T\) to 64 frames. LughaLab treats this as an
empirical reconstruction problem rather than assuming a resampling strategy.
Candidate temporal transformations are evaluated against the complete clean
124-sample test split to determine which most closely reproduces checkpoint
behaviour.

### Temporal Reconstruction Result

Five candidate temporal preprocessing strategies were evaluated on all 124 clean
KSL test samples.

Four strategies — linear interpolation, nearest-neighbour resampling,
floor-index resampling, and repeat-last-frame padding — each produced
123/124 correct predictions (99.19%).

Zero-frame padding produced only 41/124 correct predictions (33.06%).

Thus, evaluation accuracy strongly rejects naive zero-padding but does not
uniquely identify the original temporal transformation.

The small four-class task is sufficiently separable that several distinct
temporal transformations preserve almost identical classification accuracy.

Therefore, LughaLab does not claim that the highest-confidence candidate
reconstructs the original preprocessing exactly.

The remaining discrepancy between reproduced accuracy (99.19%) and the
checkpoint's reported 100% validation accuracy may reflect an undocumented
preprocessing step, evaluation-set distinction, or a non-parameter model
implementation detail such as activation function or Transformer LayerNorm
placement.

## KSL Unknown-Sign Rejection Diagnostic

The reproduced KSL classifier is a closed-set four-class model. It therefore
cannot natively represent an unknown class:

\[
\mathcal{Y}
=
\{
\text{father},
\text{hello},
\text{is},
\text{my}
\}.
\]

For any input \(x\),

\[
\sum_{c\in\mathcal{Y}} p(c\mid x)=1,
\]

including inputs outside the model's training distribution.

LughaLab therefore evaluates three descriptive uncertainty statistics:

\[
p_{\max}(x)
=
\max_c p(c\mid x),
\]

\[
M(x)
=
p_{(1)}(x)-p_{(2)}(x),
\]

and normalized entropy

\[
H_N(x)
=
\frac{
-\sum_c p_c(x)\log p_c(x)
}{
\log |\mathcal{Y}|
}.
\]

The full 124-sample clean test split is used to estimate the in-distribution
reference distribution.

Exploratory rejection thresholds are defined using the lower 5th percentile of
ID maximum probability and prediction margin, together with the upper 95th
percentile of ID normalized entropy.

Four synthetic distribution-shift controls are evaluated:

1. temporal reversal;
2. temporal frame shuffling;
3. static-frame freezing;
4. coordinate perturbation.

An external unlabelled gesture video is additionally retained as a pipeline
OOD control.

These transformations are not genuine unknown KSL lexical classes. They are
stress tests for confidence behaviour under departure from the clean training
distribution.

Consequently, rejection rates are interpreted as evidence about closed-set
model reliability rather than as validated open-set KSL recognition
performance.

## KSL Representation Ablation

Following the closed-set reliability analysis, LughaLab investigates which
components of the landmark representation drive classification.

The source representation contains 126 hand-landmark dimensions. Because the
internal ordering of the two 63-dimensional hand blocks has not yet been
independently verified from source preprocessing code, they are denoted

\[
B_1 = X_{:,0:63},
\qquad
B_2 = X_{:,63:126}.
\]

No anatomical left/right interpretation is imposed at this stage.

Feature ablation uses checkpoint training means rather than arbitrary zeros.
Because checkpoint preprocessing applies

\[
z_j =
\frac{x_j-\mu_j}{\sigma_j},
\]

replacing a raw feature by its training mean gives

\[
z_j=0.
\]

This provides a more principled neutralisation operation than raw zeroing.

For each ablation \(A\), model dependence is quantified by the change in
probability assigned to the true class:

\[
\Delta p_y
=
p_\theta(y\mid X)
-
p_\theta(y\mid A(X)).
\]

The experiment evaluates feature-block removal, feature-block isolation,
temporal-half ablation, and temporal-mean freezing.

Temporal-mean freezing replaces the complete sequence by its average landmark
configuration repeated across all 64 frames:

\[
\bar{x}
=
\frac{1}{64}
\sum_{t=1}^{64}x_t,
\qquad
X_{\mathrm{freeze}}
=
(\bar{x},\ldots,\bar{x}).
\]

This removes most explicit motion information while approximately preserving
average hand configuration.

Ablation effects are interpreted as model dependence rather than causal
linguistic importance. Modified sequences may themselves be outside the
training distribution.