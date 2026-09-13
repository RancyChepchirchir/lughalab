import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"

INPUT_PATH = RESULTS_DIR / "swahili_similarity_results.json"

SUMMARY_JSON = RESULTS_DIR / "swahili_similarity_summary.json"
SUMMARY_CSV = RESULTS_DIR / "swahili_similarity_category_summary.csv"
DISAGREEMENT_CSV = RESULTS_DIR / "swahili_similarity_disagreements.csv"


SWAHBERT = "pranaydeeps/SwahBERT-base-cased"
AFROXLMR = "Davlan/afro-xlmr-base"


def safe_mean(values):
    values = [value for value in values if value is not None]

    if not values:
        return None

    return mean(values)


def safe_median(values):
    values = [value for value in values if value is not None]

    if not values:
        return None

    return median(values)


def round_or_none(value, digits=6):
    if value is None:
        return None

    return round(value, digits)


def load_results():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def analyse():
    payload = load_results()
    records = payload["records"]

    category_data = defaultdict(
        lambda: {
            "swahbert_similarity": [],
            "afroxlmr_similarity": [],
            "similarity_gap": [],
            "absolute_similarity_gap": [],
            "swahbert_latency": [],
            "afroxlmr_latency": [],
        }
    )

    disagreements = []

    all_swahbert_similarity = []
    all_afroxlmr_similarity = []

    all_swahbert_latency = []
    all_afroxlmr_latency = []

    all_similarity_gaps = []

    for record in records:
        models = record["models"]

        swahbert = models[SWAHBERT]
        afroxlmr = models[AFROXLMR]

        swahbert_similarity = float(
            swahbert["similarity"]
        )

        afroxlmr_similarity = float(
            afroxlmr["similarity"]
        )

        swahbert_latency = float(
            swahbert["inference_time_ms"]
        )

        afroxlmr_latency = float(
            afroxlmr["inference_time_ms"]
        )

        similarity_gap = (
            swahbert_similarity
            - afroxlmr_similarity
        )

        absolute_gap = abs(
            similarity_gap
        )

        category = record["category"]

        bucket = category_data[category]

        bucket[
            "swahbert_similarity"
        ].append(
            swahbert_similarity
        )

        bucket[
            "afroxlmr_similarity"
        ].append(
            afroxlmr_similarity
        )

        bucket[
            "similarity_gap"
        ].append(
            similarity_gap
        )

        bucket[
            "absolute_similarity_gap"
        ].append(
            absolute_gap
        )

        bucket[
            "swahbert_latency"
        ].append(
            swahbert_latency
        )

        bucket[
            "afroxlmr_latency"
        ].append(
            afroxlmr_latency
        )

        all_swahbert_similarity.append(
            swahbert_similarity
        )

        all_afroxlmr_similarity.append(
            afroxlmr_similarity
        )

        all_swahbert_latency.append(
            swahbert_latency
        )

        all_afroxlmr_latency.append(
            afroxlmr_latency
        )

        all_similarity_gaps.append(
            similarity_gap
        )

        disagreements.append(
            {
                "id": record["id"],
                "category": category,
                "expected_relation": record[
                    "expected_relation"
                ],
                "text_a": record["text_a"],
                "text_b": record["text_b"],
                "swahbert_similarity": (
                    swahbert_similarity
                ),
                "afroxlmr_similarity": (
                    afroxlmr_similarity
                ),
                "similarity_gap": (
                    similarity_gap
                ),
                "absolute_similarity_gap": (
                    absolute_gap
                ),
            }
        )

    # --------------------------------------------------------
    # Category-level summary
    # --------------------------------------------------------

    category_summary = []

    for category in sorted(
        category_data.keys()
    ):
        bucket = category_data[category]

        sw_mean = safe_mean(
            bucket[
                "swahbert_similarity"
            ]
        )

        afro_mean = safe_mean(
            bucket[
                "afroxlmr_similarity"
            ]
        )

        gap_mean = safe_mean(
            bucket[
                "similarity_gap"
            ]
        )

        abs_gap_mean = safe_mean(
            bucket[
                "absolute_similarity_gap"
            ]
        )

        sw_latency_mean = safe_mean(
            bucket[
                "swahbert_latency"
            ]
        )

        afro_latency_mean = safe_mean(
            bucket[
                "afroxlmr_latency"
            ]
        )

        category_summary.append(
            {
                "category": category,
                "count": len(
                    bucket[
                        "swahbert_similarity"
                    ]
                ),
                "swahbert_mean_similarity": (
                    round_or_none(
                        sw_mean
                    )
                ),
                "afroxlmr_mean_similarity": (
                    round_or_none(
                        afro_mean
                    )
                ),
                "mean_similarity_gap": (
                    round_or_none(
                        gap_mean
                    )
                ),
                "mean_absolute_gap": (
                    round_or_none(
                        abs_gap_mean
                    )
                ),
                "swahbert_mean_latency_ms": (
                    round_or_none(
                        sw_latency_mean,
                        3,
                    )
                ),
                "afroxlmr_mean_latency_ms": (
                    round_or_none(
                        afro_latency_mean,
                        3,
                    )
                ),
            }
        )

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    global_summary = {
        "record_count": len(records),

        "swahbert": {
            "mean_similarity": round_or_none(
                safe_mean(
                    all_swahbert_similarity
                )
            ),
            "median_similarity": round_or_none(
                safe_median(
                    all_swahbert_similarity
                )
            ),
            "mean_latency_ms": round_or_none(
                safe_mean(
                    all_swahbert_latency
                ),
                3,
            ),
            "median_latency_ms": round_or_none(
                safe_median(
                    all_swahbert_latency
                ),
                3,
            ),
        },

        "afroxlmr": {
            "mean_similarity": round_or_none(
                safe_mean(
                    all_afroxlmr_similarity
                )
            ),
            "median_similarity": round_or_none(
                safe_median(
                    all_afroxlmr_similarity
                )
            ),
            "mean_latency_ms": round_or_none(
                safe_mean(
                    all_afroxlmr_latency
                ),
                3,
            ),
            "median_latency_ms": round_or_none(
                safe_median(
                    all_afroxlmr_latency
                ),
                3,
            ),
        },

        "similarity_gap": {
            "mean": round_or_none(
                safe_mean(
                    all_similarity_gaps
                )
            ),
            "median": round_or_none(
                safe_median(
                    all_similarity_gaps
                )
            ),
            "mean_absolute": round_or_none(
                safe_mean(
                    [
                        abs(value)
                        for value
                        in all_similarity_gaps
                    ]
                )
            ),
        },
    }

    # --------------------------------------------------------
    # Rank disagreements
    # --------------------------------------------------------

    disagreements.sort(
        key=lambda row: row[
            "absolute_similarity_gap"
        ],
        reverse=True,
    )

    top_disagreements = (
        disagreements[:10]
    )

    result_payload = {
        "experiment": (
            "LughaLab Swahili "
            "contextual embedding "
            "similarity analysis"
        ),
        "source_results": str(
            INPUT_PATH
        ),
        "global_summary": (
            global_summary
        ),
        "category_summary": (
            category_summary
        ),
        "largest_model_disagreements": (
            top_disagreements
        ),
        "scientific_note": (
            "These results compare raw cosine "
            "similarities produced from mean-pooled "
            "contextual embeddings. They should be "
            "treated as exploratory model-behaviour "
            "diagnostics rather than semantic "
            "accuracy measurements."
        ),
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result_payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Save category summary CSV
    # --------------------------------------------------------

    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "category",
            "count",
            "swahbert_mean_similarity",
            "afroxlmr_mean_similarity",
            "mean_similarity_gap",
            "mean_absolute_gap",
            "swahbert_mean_latency_ms",
            "afroxlmr_mean_latency_ms",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            category_summary
        )

    # --------------------------------------------------------
    # Save disagreement CSV
    # --------------------------------------------------------

    with DISAGREEMENT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "id",
            "category",
            "expected_relation",
            "swahbert_similarity",
            "afroxlmr_similarity",
            "similarity_gap",
            "absolute_similarity_gap",
            "text_a",
            "text_b",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            disagreements
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "LughaLab Swahili Similarity Analysis"
    )
    print("=" * 78)
    print()

    print("GLOBAL")
    print("-" * 78)

    print(
        "SwahBERT mean similarity: "
        f"{global_summary['swahbert']['mean_similarity']}"
    )

    print(
        "AfroXLM-R mean similarity: "
        f"{global_summary['afroxlmr']['mean_similarity']}"
    )

    print(
        "Mean model gap: "
        f"{global_summary['similarity_gap']['mean']}"
    )

    print(
        "Mean absolute model gap: "
        f"{global_summary['similarity_gap']['mean_absolute']}"
    )

    print(
        "SwahBERT mean latency: "
        f"{global_summary['swahbert']['mean_latency_ms']} ms"
    )

    print(
        "AfroXLM-R mean latency: "
        f"{global_summary['afroxlmr']['mean_latency_ms']} ms"
    )

    print()
    print("CATEGORY SUMMARY")
    print("-" * 78)

    for row in category_summary:
        print(
            f"{row['category']:<20} "
            f"SwahBERT={row['swahbert_mean_similarity']!s:<10} "
            f"AfroXLM-R={row['afroxlmr_mean_similarity']!s:<10} "
            f"|Δ|={row['mean_absolute_gap']}"
        )

    print()
    print(
        "TOP MODEL DISAGREEMENTS"
    )
    print("-" * 78)

    for row in top_disagreements[:5]:
        print(
            f"{row['id']:<15} "
            f"{row['category']:<18} "
            f"Δ={row['similarity_gap']:.6f} "
            f"|Δ|={row['absolute_similarity_gap']:.6f}"
        )

    print()
    print("-" * 78)
    print(
        f"Saved JSON: {SUMMARY_JSON}"
    )
    print(
        f"Saved summary CSV: {SUMMARY_CSV}"
    )
    print(
        f"Saved disagreements CSV: {DISAGREEMENT_CSV}"
    )
    print("-" * 78)
    print()


if __name__ == "__main__":
    analyse()