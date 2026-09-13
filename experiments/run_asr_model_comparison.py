import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from jiwer import cer, wer


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

sys.path.insert(
    0,
    str(BACKEND),
)

from app.services.swahili_asr import (
    transcribe_swahili,
)


MANIFEST_PATH = (
    ROOT
    / "data"
    / "speech"
    / "manifest.json"
)

RESULTS_DIR = (
    ROOT
    / "experiments"
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "swahili_asr_model_comparison.json"
)


MODELS = {
    "whisper-small": (
        "openai/whisper-small"
    ),
    "sauti-v1": (
        "Finiflowlabs/sauti-asr-v1"
    ),
}


def load_manifest():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalise_text(
    text: str,
) -> str:
    text = (
        text
        .lower()
        .strip()
    )

    text = re.sub(
        r"[^\w\s'-]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def round_value(
    value,
):
    return round(
        float(value),
        6,
    )


def main():
    manifest = load_manifest()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    model_scores = defaultdict(
        lambda: {
            "wer": [],
            "cer": [],
            "latency_ms": [],
            "audio_seconds": [],
        }
    )

    category_scores = defaultdict(
        lambda: defaultdict(
            lambda: {
                "wer": [],
                "cer": [],
            }
        )
    )

    print()
    print("=" * 80)
    print(
        "LughaLab Swahili ASR Model Comparison"
    )
    print("=" * 80)
    print()

    experiment_start = (
        time.perf_counter()
    )

    for index, item in enumerate(
        manifest,
        start=1,
    ):
        audio_path = (
            ROOT
            / item["audio"]
        )

        print(
            f"[{index:02d}/{len(manifest):02d}] "
            f"{item['id']:<16} "
            f"{item['category']}"
        )

        reference_raw = (
            item["reference"]
        )

        reference = normalise_text(
            reference_raw
        )

        model_results = {}

        for short_name, model_name in (
            MODELS.items()
        ):
            print(
                f"    → {short_name}"
            )

            result = (
                transcribe_swahili(
                    str(audio_path),
                    model_name=model_name,
                )
            )

            hypothesis_raw = (
                result[
                    "transcription"
                ]
            )

            hypothesis = (
                normalise_text(
                    hypothesis_raw
                )
            )

            sample_wer = wer(
                reference,
                hypothesis,
            )

            sample_cer = cer(
                reference,
                hypothesis,
            )

            model_results[
                short_name
            ] = {
                "model": (
                    model_name
                ),

                "hypothesis_raw": (
                    hypothesis_raw
                ),

                "hypothesis_normalised": (
                    hypothesis
                ),

                "wer": (
                    round_value(
                        sample_wer
                    )
                ),

                "cer": (
                    round_value(
                        sample_cer
                    )
                ),

                "duration_seconds": (
                    result[
                        "duration_seconds"
                    ]
                ),

                "inference_time_ms": (
                    result[
                        "inference_time_ms"
                    ]
                ),

                "device": (
                    result[
                        "device"
                    ]
                ),
            }

            model_scores[
                short_name
            ]["wer"].append(
                sample_wer
            )

            model_scores[
                short_name
            ]["cer"].append(
                sample_cer
            )

            model_scores[
                short_name
            ][
                "latency_ms"
            ].append(
                result[
                    "inference_time_ms"
                ]
            )

            model_scores[
                short_name
            ][
                "audio_seconds"
            ].append(
                result[
                    "duration_seconds"
                ]
            )

            category_scores[
                item["category"]
            ][
                short_name
            ]["wer"].append(
                sample_wer
            )

            category_scores[
                item["category"]
            ][
                short_name
            ]["cer"].append(
                sample_cer
            )

        whisper_wer = (
            model_results[
                "whisper-small"
            ]["wer"]
        )

        sauti_wer = (
            model_results[
                "sauti-v1"
            ]["wer"]
        )

        whisper_cer = (
            model_results[
                "whisper-small"
            ]["cer"]
        )

        sauti_cer = (
            model_results[
                "sauti-v1"
            ]["cer"]
        )

        records.append(
            {
                "id": item["id"],
                "category": (
                    item[
                        "category"
                    ]
                ),
                "audio": (
                    item[
                        "audio"
                    ]
                ),
                "reference_raw": (
                    reference_raw
                ),
                "reference_normalised": (
                    reference
                ),
                "models": (
                    model_results
                ),
                "wer_improvement": (
                    round_value(
                        whisper_wer
                        - sauti_wer
                    )
                ),
                "cer_improvement": (
                    round_value(
                        whisper_cer
                        - sauti_cer
                    )
                ),
            }
        )

    model_summary = {}

    for model_name, values in (
        model_scores.items()
    ):
        total_audio = sum(
            values[
                "audio_seconds"
            ]
        )

        total_inference_seconds = (
            sum(
                values[
                    "latency_ms"
                ]
            )
            / 1000.0
        )

        rtf = (
            total_inference_seconds
            / total_audio
            if total_audio > 0
            else None
        )

        model_summary[
            model_name
        ] = {
            "mean_wer": (
                round_value(
                    mean(
                        values[
                            "wer"
                        ]
                    )
                )
            ),

            "mean_cer": (
                round_value(
                    mean(
                        values[
                            "cer"
                        ]
                    )
                )
            ),

            "mean_latency_ms": (
                round_value(
                    mean(
                        values[
                            "latency_ms"
                        ]
                    )
                )
            ),

            "realtime_factor": (
                round_value(
                    rtf
                )
                if rtf is not None
                else None
            ),
        }

    category_summary = {}

    for category, models in sorted(
        category_scores.items()
    ):
        category_summary[
            category
        ] = {}

        for model_name, values in (
            models.items()
        ):
            category_summary[
                category
            ][
                model_name
            ] = {
                "mean_wer": (
                    round_value(
                        mean(
                            values[
                                "wer"
                            ]
                        )
                    )
                ),

                "mean_cer": (
                    round_value(
                        mean(
                            values[
                                "cer"
                            ]
                        )
                    )
                ),
            }

    ranked_improvements = sorted(
        records,
        key=lambda row: (
            row[
                "wer_improvement"
            ]
        ),
        reverse=True,
    )

    elapsed = (
        time.perf_counter()
        - experiment_start
    )

    payload = {
        "experiment": (
            "LughaLab Swahili "
            "ASR Model Comparison v1"
        ),

        "models": MODELS,

        "dataset_size": (
            len(records)
        ),

        "model_summary": (
            model_summary
        ),

        "category_summary": (
            category_summary
        ),

        "records": records,

        "largest_sauti_improvements": (
            ranked_improvements
        ),

        "wall_time_seconds": (
            round_value(
                elapsed
            )
        ),

        "scientific_note": (
            "This experiment compares "
            "two ASR systems on the same "
            "small LughaLab recording set. "
            "Differences are local diagnostic "
            "results and should not be "
            "interpreted as population-level "
            "Swahili ASR performance."
        ),
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("-" * 80)
    print(
        "MODEL SUMMARY"
    )
    print("-" * 80)
    print()

    for model_name, row in (
        model_summary.items()
    ):
        print(
            f"{model_name:<16} "
            f"WER={row['mean_wer']:<10} "
            f"CER={row['mean_cer']:<10} "
            f"RTF={row['realtime_factor']}"
        )

    print()
    print(
        "PER-SAMPLE WER CHANGE "
        "(positive = Sauti lower WER)"
    )
    print("-" * 80)

    for row in ranked_improvements:
        print(
            f"{row['id']:<16} "
            f"{row['category']:<18} "
            f"ΔWER={row['wer_improvement']:+.6f}"
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print()


if __name__ == "__main__":
    main()