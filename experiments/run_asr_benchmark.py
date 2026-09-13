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

sys.path.insert(0, str(BACKEND))

from app.services.swahili_asr import transcribe_swahili


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
    / "swahili_asr_benchmark_results.json"
)


def load_manifest():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalise_text(text: str) -> str:
    text = text.lower().strip()

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


def round_value(value):
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

    category_scores = defaultdict(
        lambda: {
            "wer": [],
            "cer": [],
            "latency": [],
            "duration": [],
        }
    )

    print()
    print("=" * 78)
    print("LughaLab Swahili ASR Benchmark v1")
    print("=" * 78)
    print()

    total_start = time.perf_counter()

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

        result = transcribe_swahili(
            str(audio_path)
        )

        reference_raw = (
            item["reference"]
        )

        hypothesis_raw = (
            result["transcription"]
        )

        reference = normalise_text(
            reference_raw
        )

        hypothesis = normalise_text(
            hypothesis_raw
        )

        sample_wer = wer(
            reference,
            hypothesis,
        )

        sample_cer = cer(
            reference,
            hypothesis,
        )

        record = {
            "id": item["id"],
            "category": (
                item["category"]
            ),
            "audio": item["audio"],

            "reference_raw": (
                reference_raw
            ),

            "hypothesis_raw": (
                hypothesis_raw
            ),

            "reference_normalised": (
                reference
            ),

            "hypothesis_normalised": (
                hypothesis
            ),

            "wer": round_value(
                sample_wer
            ),

            "cer": round_value(
                sample_cer
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

            "model": (
                result["model"]
            ),

            "device": (
                result["device"]
            ),
        }

        records.append(
            record
        )

        bucket = category_scores[
            item["category"]
        ]

        bucket["wer"].append(
            sample_wer
        )

        bucket["cer"].append(
            sample_cer
        )

        bucket["latency"].append(
            result[
                "inference_time_ms"
            ]
        )

        bucket["duration"].append(
            result[
                "duration_seconds"
            ]
        )

    category_summary = {}

    for category, values in sorted(
        category_scores.items()
    ):
        mean_duration = mean(
            values["duration"]
        )

        mean_latency = mean(
            values["latency"]
        )

        realtime_factor = (
            (mean_latency / 1000.0)
            / mean_duration
            if mean_duration > 0
            else None
        )

        category_summary[
            category
        ] = {
            "count": len(
                values["wer"]
            ),

            "mean_wer": (
                round_value(
                    mean(
                        values["wer"]
                    )
                )
            ),

            "mean_cer": (
                round_value(
                    mean(
                        values["cer"]
                    )
                )
            ),

            "mean_latency_ms": (
                round_value(
                    mean_latency
                )
            ),

            "mean_duration_seconds": (
                round_value(
                    mean_duration
                )
            ),

            "realtime_factor": (
                round_value(
                    realtime_factor
                )
                if realtime_factor
                is not None
                else None
            ),
        }

    all_wer = [
        row["wer"]
        for row in records
    ]

    all_cer = [
        row["cer"]
        for row in records
    ]

    total_audio_seconds = sum(
        row["duration_seconds"]
        for row in records
    )

    total_inference_seconds = (
        sum(
            row["inference_time_ms"]
            for row in records
        )
        / 1000.0
    )

    global_rtf = (
        total_inference_seconds
        / total_audio_seconds
        if total_audio_seconds > 0
        else None
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    payload = {
        "experiment": (
            "LughaLab Swahili "
            "ASR Benchmark v1"
        ),

        "dataset_size": (
            len(records)
        ),

        "model": (
            records[0]["model"]
            if records
            else None
        ),

        "global_summary": {
            "mean_wer": (
                round_value(
                    mean(all_wer)
                )
                if all_wer
                else None
            ),

            "mean_cer": (
                round_value(
                    mean(all_cer)
                )
                if all_cer
                else None
            ),

            "total_audio_seconds": (
                round_value(
                    total_audio_seconds
                )
            ),

            "total_inference_seconds": (
                round_value(
                    total_inference_seconds
                )
            ),

            "realtime_factor": (
                round_value(
                    global_rtf
                )
                if global_rtf
                is not None
                else None
            ),

            "benchmark_wall_time_seconds": (
                round_value(
                    total_elapsed
                )
            ),
        },

        "category_summary": (
            category_summary
        ),

        "records": records,

        "scientific_note": (
            "WER and CER are computed "
            "against manually supplied "
            "reference transcripts after "
            "simple lowercase and punctuation "
            "normalisation. This is a small "
            "diagnostic benchmark and not a "
            "population-level estimate of "
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
    print("-" * 78)
    print("GLOBAL ASR SUMMARY")
    print("-" * 78)

    print(
        "Mean WER: ",
        payload[
            "global_summary"
        ][
            "mean_wer"
        ],
    )

    print(
        "Mean CER: ",
        payload[
            "global_summary"
        ][
            "mean_cer"
        ],
    )

    print(
        "Real-time factor: ",
        payload[
            "global_summary"
        ][
            "realtime_factor"
        ],
    )

    print()
    print("CATEGORY SUMMARY")
    print("-" * 78)

    for category, row in (
        category_summary.items()
    ):
        print(
            f"{category:<18} "
            f"WER={row['mean_wer']:<10} "
            f"CER={row['mean_cer']:<10} "
            f"RTF={row['realtime_factor']}"
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print()


if __name__ == "__main__":
    main()