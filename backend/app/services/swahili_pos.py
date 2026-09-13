import time
from functools import lru_cache
from typing import Any, Dict, List

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)


MODEL_NAME = "masakhane/swahili-pos-tagger-afroxlmr"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_pos_model():
    """
    Load the Swahili POS model once and reuse it.
    """

    device = _get_device()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def _normalise_label(label: str) -> str:
    """
    Keep model labels readable while preserving the
    task-specific label semantics supplied by the model.
    """

    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]

    return label


def tag_swahili_pos(text: str) -> Dict[str, Any]:
    tokenizer, model, device = load_pos_model()

    clean_text = text.strip()

    if not clean_text:
        raise ValueError("Text cannot be empty.")

    # --------------------------------------------------------
    # Split into words first.
    #
    # XLM-R may divide an individual word into multiple
    # subword pieces. Passing is_split_into_words=True lets us
    # recover the mapping from subwords to original words.
    # --------------------------------------------------------

    words = clean_text.split()

    start = time.perf_counter()

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    word_ids = encoded.word_ids(
        batch_index=0
    )

    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**model_inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    predictions = torch.argmax(
        probabilities,
        dim=-1,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    id2label = model.config.id2label

    # --------------------------------------------------------
    # Collapse subword predictions back to one POS prediction
    # per original whitespace-delimited word.
    #
    # We use the first subword representation for each word.
    # --------------------------------------------------------

    tagged_words: List[Dict[str, Any]] = []

    previous_word_id = None

    for token_index, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        if word_id == previous_word_id:
            continue

        label_id = int(
            predictions[token_index].item()
        )

        confidence = float(
            probabilities[
                token_index,
                label_id,
            ].item()
        )

        label = id2label.get(
            label_id,
            str(label_id),
        )

        tagged_words.append(
            {
                "token": words[word_id],
                "tag": _normalise_label(label),
                "confidence": round(
                    confidence,
                    4,
                ),
            }
        )

        previous_word_id = word_id

    return {
        "text": clean_text,
        "language": "sw",
        "model": MODEL_NAME,
        "device": str(device),
        "tokens": tagged_words,
        "token_count": len(tagged_words),
        "inference_time_ms": round(
            elapsed_ms,
            3,
        ),
        "note": (
            "Predictions are produced by a task-specific "
            "AfroXLM-R Swahili POS model. Confidence is the "
            "model softmax probability for the selected tag "
            "and should not be interpreted as calibrated "
            "probability of linguistic correctness."
        ),
    }