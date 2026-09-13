import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"

INPUT_PATH = (
    RESULTS_DIR
    / "swahili_similarity_results.json"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "swahili_semantic_separation.json"
)

SWAHBERT = "pranaydeeps/SwahBERT-base-cased"
AFROXLMR = "Davlan/afro-xlmr-base"


SIMILAR_GROUPS = {
    "paraphrase",
    "informal",
    "code_switching",
}

RELATED_GROUPS = {
    "topical_similarity",
}

CHALLENGE_GROUPS = {
    "contrast",
}

UNRELATED_GROUPS = {
    "unrelated",
}


def load_results():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def safe_mean(values):
    return (
        mean(values)
        if values
        else None
    )


def safe_median(values):
    return (
        median(values)
        if values
        else None
    )


def round_value(value):
    if value is None:
        return None

    return round(value, 6)


def collect_scores(records, model_name):
    groups = defaultdict(list)

    pair_rows = []

    for record in records:
        category = record["category"]

        score = float(
            record["models"][
                model_name
            ]["similarity"]
        )

        groups[category].append(score)

        pair_rows.append(
            {
                "id": record["id"],
                "category": category,
                "expected_relation": (
                    record[
                        "expected_relation"
                    ]
                ),
                "similarity": score,
                "text_a": record[
                    "text_a"
                ],
                "text_b": record[
                    "text_b"
                ],
            }
        )

    return groups, pair_rows


def combine_categories(
    groups,
    category_names,
):
    values = []

    for category in category_names:
        values.extend(
            groups.get(
                category,
                [],
            )
        )

    return values


def summarise_model(
    records,
    model_name,
):
    groups, pair_rows = (
        collect_scores(
            records,
            model_name,
        )
    )

    similar = combine_categories(
        groups,
        SIMILAR_GROUPS,
    )

    related = combine_categories(
        groups,
        RELATED_GROUPS,
    )

    challenge = combine_categories(
        groups,
        CHALLENGE_GROUPS,
    )

    unrelated = combine_categories(
        groups,
        UNRELATED_GROUPS,
    )

    similar_mean = safe_mean(
        similar
    )

    unrelated_mean = safe_mean(
        unrelated
    )

    challenge_mean = safe_mean(
        challenge
    )

    related_mean = safe_mean(
        related
    )

    separation = None

    if (
        similar_mean is not None
        and unrelated_mean is not None
    ):
        separation = (
            similar_mean
            - unrelated_mean
        )

    contrast_penalty = None

    if (
        similar_mean is not None
        and challenge_mean is not None
    ):
        contrast_penalty = (
            similar_mean
            - challenge_mean
        )

    category_summary = {}

    for category, values in (
        sorted(
            groups.items()
        )
    ):
        category_summary[
            category
        ] = {
            "count": len(values),
            "mean": round_value(
                safe_mean(values)
            ),
            "median": round_value(
                safe_median(values)
            ),
            "minimum": round_value(
                min(values)
            ),
            "maximum": round_value(
                max(values)
            ),
        }

    return {
        "model": model_name,

        "similar_group": {
            "categories": sorted(
                SIMILAR_GROUPS
            ),
            "count": len(similar),
            "mean_similarity": (
                round_value(
                    similar_mean
                )
            ),
        },

        "related_group": {
            "categories": sorted(
                RELATED_GROUPS
            ),
            "count": len(related),
            "mean_similarity": (
                round_value(
                    related_mean
                )
            ),
        },

        "contrast_group": {
            "categories": sorted(
                CHALLENGE_GROUPS
            ),
            "count": len(challenge),
            "mean_similarity": (
                round_value(
                    challenge_mean
                )
            ),
        },

        "unrelated_group": {
            "categories": sorted(
                UNRELATED_GROUPS
            ),
            "count": len(unrelated),
            "mean_similarity": (
                round_value(
                    unrelated_mean
                )
            ),
        },

        "semantic_separation": (
            round_value(
                separation
            )
        ),

        "contrast_separation": (
            round_value(
                contrast_penalty
            )
        ),

        "category_summary": (
            category_summary
        ),

        "pairs": pair_rows,
    }


def main():
    payload = load_results()
    records = payload["records"]

    swahbert = summarise_model(
        records,
        SWAHBERT,
    )

    afroxlmr = summarise_model(
        records,
        AFROXLMR,
    )

    separation_gap = (
        swahbert[
            "semantic_separation"
        ]
        - afroxlmr[
            "semantic_separation"
        ]
    )

    contrast_gap = (
        swahbert[
            "contrast_separation"
        ]
        - afroxlmr[
            "contrast_separation"
        ]
    )

    result = {
        "experiment": (
            "LughaLab semantic "
            "separation diagnostic"
        ),

        "definition": {
            "semantic_separation": (
                "mean(similar) - "
                "mean(unrelated)"
            ),

            "contrast_separation": (
                "mean(similar) - "
                "mean(contrast)"
            ),
        },

        "models": {
            "swahbert": swahbert,
            "afroxlmr": afroxlmr,
        },

        "comparison": {
            "semantic_separation_gap": (
                round(
                    separation_gap,
                    6,
                )
            ),

            "contrast_separation_gap": (
                round(
                    contrast_gap,
                    6,
                )
            ),
        },

        "scientific_note": (
            "This experiment measures "
            "relative separation in the "
            "embedding space. It does not "
            "constitute a validated semantic "
            "textual similarity benchmark."
        ),
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 78)
    print(
        "LughaLab Semantic Separation"
    )
    print("=" * 78)
    print()

    for label, result_row in [
        (
            "SwahBERT",
            swahbert,
        ),
        (
            "AfroXLM-R",
            afroxlmr,
        ),
    ]:
        print(label)
        print("-" * 78)

        print(
            "Similar mean:      ",
            result_row[
                "similar_group"
            ][
                "mean_similarity"
            ],
        )

        print(
            "Related mean:      ",
            result_row[
                "related_group"
            ][
                "mean_similarity"
            ],
        )

        print(
            "Contrast mean:     ",
            result_row[
                "contrast_group"
            ][
                "mean_similarity"
            ],
        )

        print(
            "Unrelated mean:    ",
            result_row[
                "unrelated_group"
            ][
                "mean_similarity"
            ],
        )

        print(
            "Semantic separation:",
            result_row[
                "semantic_separation"
            ],
        )

        print(
            "Contrast separation:",
            result_row[
                "contrast_separation"
            ],
        )

        print()

    print(
        "Δ semantic separation:"
    )

    print(
        result[
            "comparison"
        ][
            "semantic_separation_gap"
        ]
    )

    print()

    print(
        "Δ contrast separation:"
    )

    print(
        result[
            "comparison"
        ][
            "contrast_separation_gap"
        ]
    )

    print()

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()


if __name__ == "__main__":
    main()