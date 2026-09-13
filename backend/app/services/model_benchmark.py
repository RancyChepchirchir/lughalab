import time
from functools import lru_cache
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MODELS = {
    "swahbert": "pranaydeeps/SwahBERT-base-cased",
    "afroxlmr": "Davlan/afro-xlmr-base",
}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


@lru_cache(maxsize=None)
def _load_model(model_name: str):
    device = _get_device()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModel.from_pretrained(
        model_name
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def _encode_sentence(
    text: str,
    model_name: str,
):
    tokenizer, model, device = _load_model(
        model_name
    )

    encoded = tokenizer(
        text,
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
    attention_mask = model_inputs[
        "attention_mask"
    ]

    expanded_mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(hidden_states.size())
        .float()
    )

    summed = torch.sum(
        hidden_states * expanded_mask,
        dim=1,
    )

    counts = torch.clamp(
        expanded_mask.sum(dim=1),
        min=1e-9,
    )

    embedding = summed / counts

    return embedding[0], device


def benchmark_models(
    text_a: str,
    text_b: str,
) -> Dict[str, Any]:

    clean_a = text_a.strip()
    clean_b = text_b.strip()

    if not clean_a or not clean_b:
        raise ValueError(
            "Both texts must be non-empty."
        )

    results: List[Dict[str, Any]] = []

    for model_key, model_name in MODELS.items():
        start = time.perf_counter()

        emb_a, device = _encode_sentence(
            clean_a,
            model_name,
        )

        emb_b, _ = _encode_sentence(
            clean_b,
            model_name,
        )

        similarity = F.cosine_similarity(
            emb_a.unsqueeze(0),
            emb_b.unsqueeze(0),
        ).item()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        results.append(
            {
                "model": model_name,
                "device": str(device),
                "embedding_dimension": int(
                    emb_a.shape[0]
                ),
                "similarity": round(
                    similarity,
                    6,
                ),
                "inference_time_ms": round(
                    elapsed_ms,
                    3,
                ),
            }
        )

    fastest = min(
        results,
        key=lambda row: row[
            "inference_time_ms"
        ],
    )

    highest_similarity = max(
        results,
        key=lambda row: row[
            "similarity"
        ],
    )

    return {
        "text_a": clean_a,
        "text_b": clean_b,
        "language": "sw",
        "results": results,
        "fastest_model": fastest["model"],
        "highest_similarity_model": (
            highest_similarity["model"]
        ),
        "note": (
            "Cosine similarity is computed from "
            "mean-pooled contextual embeddings. "
            "These encoder representations are not "
            "sentence-transformer embeddings, so higher "
            "similarity should not be treated as proof "
            "of better semantic quality."
        ),
    }