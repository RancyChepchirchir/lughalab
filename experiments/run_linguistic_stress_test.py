import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

sys.path.insert(
    0,
    str(BACKEND),
)

from app.services.model_benchmark import (
    benchmark_models,
)


DATA_PATH = (
    ROOT
    / "data"
    / "swahili_linguistic_stress_test.json"
)

RESULTS_DIR = (
    ROOT
    / "experiments"
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "swahili_linguistic_stress_test_results.json"
)


SWAHBERT = (
    "pranaydeeps/SwahBERT-base-cased"
)

AFROXLMR = (
    "Davlan/afro-xlmr-base"
)


def load_dataset():
    with DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def round_value(value):
    return round(
        float(value),
        6,
    )


def run():
    dataset = load_dataset()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    phenomenon_scores = defaultdict(
        lambda: {
            SWAHBERT: [],
            AFROXLMR: [],
        }
    )

    print()
    print("=" * 78)
    print(
        "LughaLab Swahili Linguistic Stress Test"
    )
    print("=" * 78)
    print()

    start = time.perf_counter()

    for index, item in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(dataset):02d}] "
            f"{item['id']:<12} "
            f"{item['phenomenon']}"
        )

        result = benchmark_models(
            item["text_a"],
            item["text_b"],
        )

        model_results = {}

        for model_result in result[
            "results"
        ]:
            model_name = (
                model_result["model"]
            )

            similarity = float(
                model_result[
                    "similarity"
                ]
            )

            representational_shift = (
                1.0 - similarity
            )

            model_results[
                model_name
            ] = {
                "similarity": (
                    round_value(
                        similarity
                    )
                ),
                "representational_shift": (
                    round_value(
                        representational_shift
                    )
                ),
                "inference_time_ms": (
                    model_result[
                        "inference_time_ms"
                    ]
                ),
            }

            phenomenon_scores[
                item["phenomenon"]
            ][model_name].append(
                representational_shift
            )

        records.append(
            {
                "id": item["id"],
                "phenomenon": (
                    item["phenomenon"]
                ),
                "change": item["change"],
                "semantic_effect": (
                    item[
                        "semantic_effect"
                    ]
                ),
                "description": (
                    item["description"]
                ),
                "text_a": item["text_a"],
                "text_b": item["text_b"],
                "models": model_results,
            }
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    phenomenon_summary = {}

    for phenomenon, model_values in sorted(
        phenomenon_scores.items()
    ):
        phenomenon_summary[
            phenomenon
        ] = {}

        for model_name, values in (
            model_values.items()
        ):
            phenomenon_summary[
                phenomenon
            ][model_name] = {
                "count": len(values),
                "mean_representational_shift": (
                    round_value(
                        mean(values)
                    )
                ),
                "minimum_shift": (
                    round_value(
                        min(values)
                    )
                ),
                "maximum_shift": (
                    round_value(
                        max(values)
                    )
                ),
            }

    payload = {
        "experiment": (
            "LughaLab Swahili "
            "Linguistic Stress Test v1"
        ),

        "definition": {
            "representational_shift": (
                "1 - cosine_similarity"
            )
        },

        "dataset_size": len(
            records
        ),

        "runtime_seconds": round(
            elapsed,
            3,
        ),

        "phenomenon_summary": (
            phenomenon_summary
        ),

        "records": records,

        "scientific_note": (
            "Representational shift measures "
            "movement in embedding space after "
            "a controlled linguistic transformation. "
            "Larger movement is not inherently better: "
            "the interpretation depends on whether "
            "the transformation is intended to preserve "
            "or alter meaning."
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
    print(
        "MEAN REPRESENTATIONAL SHIFT"
    )
    print("-" * 78)
    print()

    for phenomenon, models in (
        phenomenon_summary.items()
    ):
        sw = models[
            SWAHBERT
        ][
            "mean_representational_shift"
        ]

        afro = models[
            AFROXLMR
        ][
            "mean_representational_shift"
        ]

        print(
            f"{phenomenon:<22} "
            f"SwahBERT={sw:<10} "
            f"AfroXLM-R={afro}"
        )

    print()
    print("-" * 78)

    print(
        f"Runtime: {elapsed:.3f}s"
    )

    print(
        f"Saved:   {OUTPUT_PATH}"
    )

    print("-" * 78)
    print()


if __name__ == "__main__":
    run()