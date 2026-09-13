import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import librosa
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)


TARGET_SAMPLE_RATE = 16000

DEFAULT_MODEL = "openai/whisper-small"

SUPPORTED_MODELS = {
    "whisper-small": "openai/whisper-small",
    "sauti-v1": "Finiflowlabs/sauti-asr-v1",
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


def _get_dtype(
    device: torch.device,
) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16

    return torch.float32


@lru_cache(maxsize=None)
def _load_model(
    model_name: str,
):
    device = _get_device()
    dtype = _get_dtype(device)

    processor = AutoProcessor.from_pretrained(
        model_name
    )

    model = (
        AutoModelForSpeechSeq2Seq
        .from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    )

    model.to(device)
    model.eval()

    return (
        processor,
        model,
        device,
        dtype,
    )


def _load_audio(
    audio_path: str,
):
    waveform, sampling_rate = (
        librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
        )
    )

    return (
        waveform,
        sampling_rate,
    )


def transcribe_swahili(
    audio_path: str,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    path = Path(audio_path)

    if not path.exists():
        raise ValueError(
            f"Audio file does not exist: {path}"
        )

    (
        processor,
        model,
        device,
        dtype,
    ) = _load_model(
        model_name
    )

    waveform, sampling_rate = (
        _load_audio(
            str(path)
        )
    )

    duration_seconds = (
        len(waveform)
        / sampling_rate
    )

    inputs = processor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )

    input_features = (
        inputs.input_features
        .to(
            device=device,
            dtype=dtype,
        )
    )

    generation_kwargs = {}

    try:
        forced_decoder_ids = (
            processor
            .get_decoder_prompt_ids(
                language="swahili",
                task="transcribe",
            )
        )

        generation_kwargs[
            "forced_decoder_ids"
        ] = forced_decoder_ids

    except Exception:
        pass

    start = time.perf_counter()

    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features,
            **generation_kwargs,
        )

    inference_time_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    transcription = (
        processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]
        .strip()
    )

    return {
        "model": model_name,
        "language": "sw",
        "device": str(device),
        "sampling_rate": (
            sampling_rate
        ),
        "duration_seconds": round(
            duration_seconds,
            3,
        ),
        "transcription": (
            transcription
        ),
        "inference_time_ms": round(
            inference_time_ms,
            3,
        ),
        "segments": [],
        "note": (
            "Swahili ASR transcription. "
            "Accuracy should be evaluated "
            "against a human reference "
            "transcript using WER and CER."
        ),
    }