import math
import time
from functools import lru_cache
from typing import Dict, Any

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "pranaydeeps/SwahBERT-base-cased"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_encoder():
    """
    Load SwahBERT once and reuse it across requests.
    """

    device = _get_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    return tokenizer, model, device


def _l2_norm(values):
    return math.sqrt(sum(value * value for value in values))


def analyse_swahili(text: str) -> Dict[str, Any]:
    tokenizer, model, device = load_encoder()

    clean_text = text.strip()

    if not clean_text:
        raise ValueError("Text cannot be empty.")

    start = time.perf_counter()

    encoded = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**model_inputs)

    hidden_states = outputs.last_hidden_state

    # ---------------------------------------------------------
    # Sentence representation
    #
    # Instead of blindly using [CLS], use masked mean pooling
    # over contextual token representations.
    # ---------------------------------------------------------

    attention_mask = model_inputs["attention_mask"]

    expanded_mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(hidden_states.size())
        .float()
    )

    summed_embeddings = torch.sum(
        hidden_states * expanded_mask,
        dim=1,
    )

    token_counts = torch.clamp(
        expanded_mask.sum(dim=1),
        min=1e-9,
    )

    sentence_embedding = (
        summed_embeddings / token_counts
    )[0]

    sentence_embedding = (
        sentence_embedding
        .detach()
        .cpu()
        .float()
        .tolist()
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    token_ids = encoded["input_ids"][0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(
        token_ids
    )

    token_records = [
        {
            "token": token,
            "token_id": token_id,
        }
        for token, token_id in zip(
            tokens,
            token_ids,
        )
    ]

    return {
        "text": clean_text,
        "model": MODEL_NAME,
        "language": "sw",
        "device": str(device),
        "tokens": token_records,
        "token_count": len(tokens),
        "embedding_dimension": len(
            sentence_embedding
        ),
        "sentence_embedding": sentence_embedding,
        "embedding_norm": _l2_norm(
            sentence_embedding
        ),
        "inference_time_ms": round(
            elapsed_ms,
            3,
        ),
        "note": (
            "Sentence representation produced by "
            "mean pooling SwahBERT contextual embeddings. "
            "This is an encoder representation, not a "
            "task-specific classification prediction."
        ),
    }