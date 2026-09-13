import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

sys.path.insert(0, str(BACKEND))

from app.services.model_benchmark import benchmark_models


DATA_PATH = ROOT / "data" / "swahili_similarity_benchmark.json"
RESULTS_DIR = ROOT / "experiments" / "results"

JSON_OUTPUT = RESULTS_DIR / "swahili_similarity_results.json"
CSV_OUTPUT = RESULTS_DIR / "swahili_similarity_results.csv"


def load_dataset():
    with DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def run():
    dataset = load_dataset()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    print()
    print("=" * 72)
    print("LughaLab Swahili Similarity Benchmark")
    print("=" * 72)
    print()

    benchmark_start = time.perf_counter()

    for index, item in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(dataset):02d}] "
            f"{item['id']} "
            f"({item['category']})"
        )

        result = benchmark_models(
            item["text_a"],
            item["text_b"],
        )

        similarities = {}

        for model_result in result["results"]:
            model_name = model_result["model"]

            similarities[model_name] = {
                "similarity": model_result[
                    "similarity"
                ],
                "inference_time_ms": model_result[
                    "inference_time_ms"
                ],
                "embedding_dimension": model_result[
                    "embedding_dimension"
                ],
                "device": model_result[
                    "device"
                ],
            }

        record = {
            "id": item["id"],
            "category": item["category"],
            "expected_relation": item[
                "expected_relation"
            ],
            "text_a": item["text_a"],
            "text_b": item["text_b"],
            "models": similarities,
        }

        records.append(record)

    total_seconds = (
        time.perf_counter()
        - benchmark_start
    )

    payload = {
        "experiment": (
            "LughaLab Swahili contextual "
            "embedding similarity benchmark"
        ),
        "dataset_size": len(records),
        "total_runtime_seconds": round(
            total_seconds,
            3,
        ),
        "records": records,
    }

    with JSON_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    rows = []

    for record in records:
        for model_name, values in (
            record["models"].items()
        ):
            rows.append(
                {
                    "id": record["id"],
                    "category": record[
                        "category"
                    ],
                    "expected_relation": record[
                        "expected_relation"
                    ],
                    "model": model_name,
                    "similarity": values[
                        "similarity"
                    ],
                    "inference_time_ms": values[
                        "inference_time_ms"
                    ],
                    "embedding_dimension": values[
                        "embedding_dimension"
                    ],
                    "device": values[
                        "device"
                    ],
                    "text_a": record[
                        "text_a"
                    ],
                    "text_b": record[
                        "text_b"
                    ],
                }
            )

    with CSV_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "category",
                "expected_relation",
                "model",
                "similarity",
                "inference_time_ms",
                "embedding_dimension",
                "device",
                "text_a",
                "text_b",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("-" * 72)
    print(
        f"Saved JSON: {JSON_OUTPUT}"
    )
    print(
        f"Saved CSV:  {CSV_OUTPUT}"
    )
    print(
        f"Runtime:    {total_seconds:.3f}s"
    )
    print("-" * 72)
    print()


if __name__ == "__main__":
    run()