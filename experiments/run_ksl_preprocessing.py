import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

sys.path.insert(
    0,
    str(BACKEND),
)


from app.services.ksl_landmarks import (
    extract_ksl_landmarks,
    frames_to_array,
)

from app.services.ksl_preprocessing import (
    preprocess_landmarks,
)


VIDEO_DIR = (
    ROOT
    / "data"
    / "ksl"
    / "videos"
)

LANDMARK_DIR = (
    ROOT
    / "data"
    / "ksl"
    / "landmarks"
)

RESULTS_DIR = (
    ROOT
    / "experiments"
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "ksl_preprocessing_results.json"
)


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
}


def round_nested(
    value,
):
    if isinstance(
        value,
        dict,
    ):
        return {
            key: round_nested(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        float,
    ):
        return round(
            value,
            6,
        )

    return value


def main():
    videos = sorted(
        [
            path
            for path
            in VIDEO_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in VIDEO_EXTENSIONS
            )
        ]
    )

    if not videos:
        raise RuntimeError(
            "No test videos found in "
            f"{VIDEO_DIR}"
        )

    LANDMARK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    print()
    print("=" * 78)
    print(
        "LughaLab KSL "
        "Landmark Preprocessing"
    )
    print("=" * 78)
    print()

    for video in videos:
        print(
            f"Processing: {video.name}"
        )

        extraction = (
            extract_ksl_landmarks(
                str(video)
            )
        )

        sequence = (
            frames_to_array(
                extraction
            )
        )

        processed = (
            preprocess_landmarks(
                sequence
            )
        )

        output_array = (
            processed[
                "sequence"
            ]
        )

        npy_path = (
            LANDMARK_DIR
            / (
                video.stem
                + "_64x225.npy"
            )
        )

        np.save(
            npy_path,
            output_array,
        )

        record = {
            "video": (
                video.name
            ),

            "frames_read": (
                extraction[
                    "frames_read"
                ]
            ),

            "hand_detection_rate": (
                extraction[
                    "detection_rate"
                ]
            ),

            "original_shape": (
                processed[
                    "original_shape"
                ]
            ),

            "output_shape": (
                processed[
                    "output_shape"
                ]
            ),

            "missingness_before": (
                processed[
                    "missingness_before"
                ]
            ),

            "missingness_after": (
                processed[
                    "missingness_after"
                ]
            ),

            "geometry": (
                processed[
                    "geometry"
                ]
            ),

            "saved_landmarks": (
                str(
                    npy_path.relative_to(
                        ROOT
                    )
                )
            ),
        }

        records.append(
            round_nested(
                record
            )
        )

        print(
            "    raw shape: ",
            tuple(
                sequence.shape
            ),
        )

        print(
            "    model shape:",
            tuple(
                output_array.shape
            ),
        )

        print(
            "    hand detection:",
            extraction[
                "detection_rate"
            ],
        )

        print()

    payload = {
        "experiment": (
            "LughaLab KSL "
            "Preprocessing v1"
        ),

        "target_shape": [
            64,
            225,
        ],

        "records": (
            records
        ),

        "scientific_note": (
            "The videos used at this stage "
            "validate the landmark preprocessing "
            "pipeline only. They must not be "
            "treated as labelled Kenyan Sign "
            "Language observations unless their "
            "linguistic provenance has been "
            "independently established."
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

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()


if __name__ == "__main__":
    main()