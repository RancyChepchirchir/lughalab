import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "experiments" / "results"

INPUT_PATH = (
    RESULTS_DIR
    / "swahili_linguistic_stress_test_results.json"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "swahili_linguistic_sensitivity_analysis.json"
)


SWAHBERT = "pranaydeeps/SwahBERT-base-cased"
AFROXLMR = "Davlan/afro-xlmr-base"


MEANING_CHANGING = {
    "meaning_change",
    "temporal_change",
    "participant_change",
}

MEANING_PRESERVING = {
    "meaning_preserving",
    "approximately_meaning_preserving",
}


def load_results():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def round_value(value):
    if value is None:
        return None

    return round(
        float(value),
        6,
    )


def analyse_model(records, model_name):
    changing = []
    preserving = []

    changing_rows = []
    preserving_rows = []

    for row in records:
        effect = row[
            "semantic_effect"
        ]

        shift = float(
            row["models"][
                model_name
            ][
                "representational_shift"
            ]
        )

        result_row = {
            "id": row["id"],
            "phenomenon": (
                row["phenomenon"]
            ),
            "semantic_effect": effect,
            "representational_shift": (
                shift
            ),
            "text_a": row[
                "text_a"
            ],
            "text_b": row[
                "text_b"
            ],
        }

        if effect in MEANING_CHANGING:
            changing.append(
                shift
            )

            changing_rows.append(
                result_row
            )

        elif effect in MEANING_PRESERVING:
            preserving.append(
                shift
            )

            preserving_rows.append(
                result_row
            )

    changing_mean = (
        mean(changing)
        if changing
        else None
    )

    preserving_mean = (
        mean(preserving)
        if preserving
        else None
    )

    lss = None

    if (
        changing_mean is not None
        and preserving_mean is not None
    ):
        lss = (
            changing_mean
            - preserving_mean
        )

    changing_rows.sort(
        key=lambda row: (
            row[
                "representational_shift"
            ]
        )
    )

    preserving_rows.sort(
        key=lambda row: (
            row[
                "representational_shift"
            ]
        ),
        reverse=True,
    )

    return {
        "model": model_name,

        "meaning_changing": {
            "count": len(
                changing
            ),
            "mean_shift": (
                round_value(
                    changing_mean
                )
            ),
        },

        "meaning_preserving": {
            "count": len(
                preserving
            ),
            "mean_shift": (
                round_value(
                    preserving_mean
                )
            ),
        },

        "provisional_lss": (
            round_value(
                lss
            )
        ),

        "lowest_shift_meaning_changes": (
            changing_rows[:5]
        ),

        "highest_shift_meaning_preserving": (
            preserving_rows[:5]
        ),
    }


def main():
    payload = load_results()
    records = payload[
        "records"
    ]

    swahbert = analyse_model(
        records,
        SWAHBERT,
    )

    afroxlmr = analyse_model(
        records,
        AFROXLMR,
    )

    lss_difference = (
        swahbert[
            "provisional_lss"
        ]
        - afroxlmr[
            "provisional_lss"
        ]
    )

    result = {
        "experiment": (
            "LughaLab Linguistic "
            "Sensitivity vs Invariance Analysis"
        ),

        "definition": {
            "desired_sensitivity": (
                "mean representational shift "
                "for meaning-changing transformations"
            ),

            "desired_invariance": (
                "low representational shift "
                "for meaning-preserving transformations"
            ),

            "provisional_lss": (
                "mean_shift(meaning-changing) "
                "- mean_shift(meaning-preserving)"
            ),
        },

        "models": {
            "swahbert": (
                swahbert
            ),
            "afroxlmr": (
                afroxlmr
            ),
        },

        "comparison": {
            "lss_difference": (
                round_value(
                    lss_difference
                )
            )
        },

        "scientific_note": (
            "The provisional Linguistic "
            "Sensitivity-Stability score is "
            "an exploratory diagnostic only. "
            "It has not been validated as a "
            "general-purpose evaluation metric "
            "and should not be presented as a "
            "novel metric without literature "
            "review, larger datasets and "
            "human linguistic validation."
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
        "LughaLab Sensitivity vs Invariance"
    )
    print("=" * 78)
    print()

    for label, row in [
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
            "Meaning-changing mean shift: ",
            row[
                "meaning_changing"
            ][
                "mean_shift"
            ],
        )

        print(
            "Meaning-preserving mean shift:",
            row[
                "meaning_preserving"
            ][
                "mean_shift"
            ],
        )

        print(
            "Provisional LSS:              ",
            row[
                "provisional_lss"
            ],
        )

        print()

    print(
        "LSS difference "
        "(SwahBERT - AfroXLM-R):"
    )

    print(
        result[
            "comparison"
        ][
            "lss_difference"
        ]
    )

    print()

    print(
        "Potential sensitivity failures"
    )

    print("-" * 78)

    for row in (
        swahbert[
            "lowest_shift_meaning_changes"
        ]
    ):
        print(
            f"SwahBERT  "
            f"{row['id']:<12} "
            f"{row['phenomenon']:<20} "
            f"shift={row['representational_shift']:.6f}"
        )

    print()

    for row in (
        afroxlmr[
            "lowest_shift_meaning_changes"
        ]
    ):
        print(
            f"AfroXLM-R "
            f"{row['id']:<12} "
            f"{row['phenomenon']:<20} "
            f"shift={row['representational_shift']:.6f}"
        )

    print()

    print(
        "Potential invariance failures"
    )

    print("-" * 78)

    for row in (
        swahbert[
            "highest_shift_meaning_preserving"
        ]
    ):
        print(
            f"SwahBERT  "
            f"{row['id']:<12} "
            f"{row['phenomenon']:<20} "
            f"shift={row['representational_shift']:.6f}"
        )

    print()

    for row in (
        afroxlmr[
            "highest_shift_meaning_preserving"
        ]
    ):
        print(
            f"AfroXLM-R "
            f"{row['id']:<12} "
            f"{row['phenomenon']:<20} "
            f"shift={row['representational_shift']:.6f}"
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print()


if __name__ == "__main__":
    main()