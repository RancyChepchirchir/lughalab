# LughaLab

**African Language & Sign Intelligence**

LughaLab is a research-oriented platform for studying low-resource African
language technologies across text, speech, and sign language.

The project currently focuses on Swahili and Kenyan Sign Language (KSL), with
an emphasis on representation quality, model robustness, domain shift,
adaptation, and responsible evaluation rather than demo-only inference.

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
