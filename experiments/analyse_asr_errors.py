import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from jiwer import process_words


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "experiments"
    / "results"
)

INPUT_PATH = (
    RESULTS_DIR
    / "swahili_asr_model_comparison.json"
)

OUTPUT_JSON = (
    RESULTS_DIR
    / "swahili_asr_error_analysis.json"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "swahili_asr_error_analysis.csv"
)


MODELS = [
    "whisper-small",
    "sauti-v1",
]


def load_data():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def safe_rate(
    value,
    denominator,
):
    if denominator == 0:
        return 0.0

    return value / denominator


def extract_alignment(
    reference,
    hypothesis,
):
    result = process_words(
        reference,
        hypothesis,
    )

    substitutions = []
    deletions = []
    insertions = []

    ref_words = result.references[0]
    hyp_words = result.hypotheses[0]

    for chunk in result.alignments[0]:
        if chunk.type == "equal":
            continue

        ref_segment = ref_words[
            chunk.ref_start_idx:
            chunk.ref_end_idx
        ]

        hyp_segment = hyp_words[
            chunk.hyp_start_idx:
            chunk.hyp_end_idx
        ]

        if chunk.type == "substitute":
            substitutions.append(
                {
                    "reference": (
                        " ".join(
                            ref_segment
                        )
                    ),
                    "hypothesis": (
                        " ".join(
                            hyp_segment
                        )
                    ),
                }
            )

        elif chunk.type == "delete":
            deletions.append(
                {
                    "reference": (
                        " ".join(
                            ref_segment
                        )
                    )
                }
            )

        elif chunk.type == "insert":
            insertions.append(
                {
                    "hypothesis": (
                        " ".join(
                            hyp_segment
                        )
                    )
                }
            )

    return {
        "hits": result.hits,
        "substitutions_count": (
            result.substitutions
        ),
        "deletions_count": (
            result.deletions
        ),
        "insertions_count": (
            result.insertions
        ),
        "reference_word_count": (
            len(ref_words)
        ),
        "substitutions": (
            substitutions
        ),
        "deletions": (
            deletions
        ),
        "insertions": (
            insertions
        ),
    }


def main():
    data = load_data()

    records = []

    aggregate = {
        model: {
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "reference_words": 0,
            "wer": [],
            "cer": [],
        }
        for model in MODELS
    }

    substitution_pairs = {
        model: Counter()
        for model in MODELS
    }

    deleted_words = {
        model: Counter()
        for model in MODELS
    }

    inserted_words = {
        model: Counter()
        for model in MODELS
    }

    category_aggregate = defaultdict(
        lambda: {
            model: {
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "reference_words": 0,
            }
            for model in MODELS
        }
    )

    for row in data["records"]:
        analysed_models = {}

        reference = (
            row[
                "reference_normalised"
            ]
        )

        for model in MODELS:
            hypothesis = (
                row[
                    "models"
                ][
                    model
                ][
                    "hypothesis_normalised"
                ]
            )

            alignment = (
                extract_alignment(
                    reference,
                    hypothesis,
                )
            )

            model_wer = (
                row[
                    "models"
                ][
                    model
                ][
                    "wer"
                ]
            )

            model_cer = (
                row[
                    "models"
                ][
                    model
                ][
                    "cer"
                ]
            )

            analysed_models[
                model
            ] = {
                **alignment,
                "wer": model_wer,
                "cer": model_cer,
            }

            aggregate[
                model
            ]["substitutions"] += (
                alignment[
                    "substitutions_count"
                ]
            )

            aggregate[
                model
            ]["deletions"] += (
                alignment[
                    "deletions_count"
                ]
            )

            aggregate[
                model
            ]["insertions"] += (
                alignment[
                    "insertions_count"
                ]
            )

            aggregate[
                model
            ][
                "reference_words"
            ] += (
                alignment[
                    "reference_word_count"
                ]
            )

            aggregate[
                model
            ]["wer"].append(
                model_wer
            )

            aggregate[
                model
            ]["cer"].append(
                model_cer
            )

            category_bucket = (
                category_aggregate[
                    row["category"]
                ][
                    model
                ]
            )

            category_bucket[
                "substitutions"
            ] += (
                alignment[
                    "substitutions_count"
                ]
            )

            category_bucket[
                "deletions"
            ] += (
                alignment[
                    "deletions_count"
                ]
            )

            category_bucket[
                "insertions"
            ] += (
                alignment[
                    "insertions_count"
                ]
            )

            category_bucket[
                "reference_words"
            ] += (
                alignment[
                    "reference_word_count"
                ]
            )

            for item in alignment[
                "substitutions"
            ]:
                key = (
                    item["reference"],
                    item["hypothesis"],
                )

                substitution_pairs[
                    model
                ][key] += 1

            for item in alignment[
                "deletions"
            ]:
                for word in (
                    item[
                        "reference"
                    ].split()
                ):
                    deleted_words[
                        model
                    ][word] += 1

            for item in alignment[
                "insertions"
            ]:
                for word in (
                    item[
                        "hypothesis"
                    ].split()
                ):
                    inserted_words[
                        model
                    ][word] += 1

        records.append(
            {
                "id": row["id"],
                "category": (
                    row["category"]
                ),
                "reference": (
                    reference
                ),
                "models": (
                    analysed_models
                ),
            }
        )

    model_summary = {}

    for model in MODELS:
        values = aggregate[
            model
        ]

        n = values[
            "reference_words"
        ]

        substitutions = (
            values[
                "substitutions"
            ]
        )

        deletions = (
            values[
                "deletions"
            ]
        )

        insertions = (
            values[
                "insertions"
            ]
        )

        corpus_wer = safe_rate(
            substitutions
            + deletions
            + insertions,
            n,
        )

        model_summary[
            model
        ] = {
            "reference_words": n,

            "substitutions": (
                substitutions
            ),

            "deletions": (
                deletions
            ),

            "insertions": (
                insertions
            ),

            "corpus_wer": round(
                corpus_wer,
                6,
            ),

            "mean_utterance_wer": round(
                mean(
                    values[
                        "wer"
                    ]
                ),
                6,
            ),

            "mean_utterance_cer": round(
                mean(
                    values[
                        "cer"
                    ]
                ),
                6,
            ),

            "substitution_rate": round(
                safe_rate(
                    substitutions,
                    n,
                ),
                6,
            ),

            "deletion_rate": round(
                safe_rate(
                    deletions,
                    n,
                ),
                6,
            ),

            "insertion_rate": round(
                safe_rate(
                    insertions,
                    n,
                ),
                6,
            ),

            "top_substitutions": [
                {
                    "reference": ref,
                    "hypothesis": hyp,
                    "count": count,
                }
                for (
                    (
                        ref,
                        hyp,
                    ),
                    count,
                )
                in substitution_pairs[
                    model
                ].most_common(10)
            ],

            "top_deletions": [
                {
                    "word": word,
                    "count": count,
                }
                for (
                    word,
                    count,
                )
                in deleted_words[
                    model
                ].most_common(10)
            ],

            "top_insertions": [
                {
                    "word": word,
                    "count": count,
                }
                for (
                    word,
                    count,
                )
                in inserted_words[
                    model
                ].most_common(10)
            ],
        }

    category_summary = {}

    for category, models in sorted(
        category_aggregate.items()
    ):
        category_summary[
            category
        ] = {}

        for model in MODELS:
            values = (
                models[
                    model
                ]
            )

            n = (
                values[
                    "reference_words"
                ]
            )

            s = (
                values[
                    "substitutions"
                ]
            )

            d = (
                values[
                    "deletions"
                ]
            )

            i = (
                values[
                    "insertions"
                ]
            )

            category_summary[
                category
            ][
                model
            ] = {
                "substitutions": s,
                "deletions": d,
                "insertions": i,

                "corpus_wer": round(
                    safe_rate(
                        s + d + i,
                        n,
                    ),
                    6,
                ),
            }

    payload = {
        "experiment": (
            "LughaLab Swahili "
            "ASR Error Decomposition v1"
        ),

        "model_summary": (
            model_summary
        ),

        "category_summary": (
            category_summary
        ),

        "records": records,

        "scientific_note": (
            "ASR errors are decomposed into "
            "substitutions, deletions and insertions "
            "using word-level alignment. "
            "This is a small diagnostic dataset. "
            "Repeated errors should not be interpreted "
            "as general Swahili error distributions."
        ),
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    import csv

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "sample_id",
                "category",
                "model",
                "substitutions",
                "deletions",
                "insertions",
                "reference_words",
                "wer",
                "cer",
            ]
        )

        for row in records:
            for model in MODELS:
                model_row = (
                    row[
                        "models"
                    ][
                        model
                    ]
                )

                writer.writerow(
                    [
                        row["id"],
                        row["category"],
                        model,
                        model_row[
                            "substitutions_count"
                        ],
                        model_row[
                            "deletions_count"
                        ],
                        model_row[
                            "insertions_count"
                        ],
                        model_row[
                            "reference_word_count"
                        ],
                        model_row["wer"],
                        model_row["cer"],
                    ]
                )

    print()
    print("=" * 80)
    print(
        "LughaLab Swahili "
        "ASR Error Decomposition"
    )
    print("=" * 80)
    print()

    for model in MODELS:
        row = (
            model_summary[
                model
            ]
        )

        print(model)
        print("-" * 80)

        print(
            "Corpus WER:       ",
            row[
                "corpus_wer"
            ],
        )

        print(
            "Substitution rate:",
            row[
                "substitution_rate"
            ],
        )

        print(
            "Deletion rate:    ",
            row[
                "deletion_rate"
            ],
        )

        print(
            "Insertion rate:   ",
            row[
                "insertion_rate"
            ],
        )

        print()

    print(
        f"Saved: {OUTPUT_JSON}"
    )

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print()


if __name__ == "__main__":
    main()