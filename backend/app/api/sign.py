import os
import tempfile
import numpy as np

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.models.sign import (
    KSLClassificationResponse,
    LandmarkExtractionResponse,
)

from app.services.ksl_landmarks import (
    extract_ksl_landmarks,
)

from app.services.ksl_checkpoint_adapter import (
    prepare_for_legacy_checkpoint,
)

from app.services.ksl_classifier import (
    classify_checkpoint_sequence,
    extract_checkpoint_embedding,
    load_ksl_classifier,
)

from app.services.ksl_open_set import (
    evaluate_open_set,
)

from app.services.ksl_reliability import (
    assess_ksl_reliability,
)


router = APIRouter(
    prefix="/sign",
    tags=["sign"],
)


@router.post(
    "/landmarks",
    response_model=LandmarkExtractionResponse,
)
async def extract_landmarks(
    file: UploadFile = File(...),
):
    suffix = os.path.splitext(
        file.filename or ""
    )[1]

    if not suffix:
        suffix = ".mp4"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp:
            temp_path = temp.name

            contents = (
                await file.read()
            )

            temp.write(
                contents
            )

        result = (
            extract_ksl_landmarks(
                temp_path
            )
        )

        result["filename"] = (
            file.filename
            or result["filename"]
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "KSL landmark extraction "
                f"failed: {exc}"
            ),
        ) from exc

    finally:
        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):
            os.remove(
                temp_path
            )

@router.post(
    "/classify",
    response_model=KSLClassificationResponse,
)
async def classify_sign(
    file: UploadFile = File(...),
):
    suffix = os.path.splitext(
        file.filename or ""
    )[1]

    if not suffix:
        suffix = ".mp4"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp:
            temp_path = temp.name

            contents = await file.read()

            temp.write(
                contents
            )

        extraction = (
            extract_ksl_landmarks(
                temp_path
            )
        )

        frames = extraction.get(
            "frames",
            []
        )

        if not frames:
            raise ValueError(
                "No landmark frames were extracted "
                "from the uploaded video."
            )

        native_sequence = np.asarray(
            [
                frame["landmarks"]
                for frame in frames
            ],
            dtype=np.float32,
        )

        bundle = (
            load_ksl_classifier()
        )

        adapter = (
            prepare_for_legacy_checkpoint(
                native_sequence,
                mean=bundle["mean"],
                std=bundle["std"],
            )
        )

        classification = (
            classify_checkpoint_sequence(
                adapter["sequence"]
            )
        )

        embedding = (
            extract_checkpoint_embedding(
                adapter["sequence"]
            )
        )

        open_set = (
            evaluate_open_set(
                embedding
            )
        )

        reliability = (
            assess_ksl_reliability(
                detection_rate=extraction[
                    "detection_rate"
                ],
                predictions=classification[
                    "predictions"
                ],
            )
        )

        return {
            "filename": (
                file.filename
                or extraction["filename"]
            ),

            "predicted_label": (
                classification[
                    "predicted_label"
                ]
            ),

            "predicted_probability": (
                classification[
                    "predicted_probability"
                ]
            ),

            "predictions": (
                classification[
                    "predictions"
                ]
            ),

            "reliability": reliability,

            "open_set": open_set,

            "model": (
                classification["model"]
            ),

            "device": (
                classification["device"]
            ),

            "closed_set": (
                classification[
                    "closed_set"
                ]
            ),

            "supported_labels": (
                classification[
                    "supported_labels"
                ]
            ),

            "frames_read": (
                extraction[
                    "frames_read"
                ]
            ),

            "frames_detected": (
                extraction[
                    "frames_detected"
                ]
            ),

            "detection_rate": (
                extraction[
                    "detection_rate"
                ]
            ),

            "adapter": {
                "native_shape": (
                    adapter[
                        "native_shape"
                    ]
                ),

                "hands_shape": (
                    adapter[
                        "hands_shape"
                    ]
                ),

                "checkpoint_shape": (
                    adapter[
                        "checkpoint_shape"
                    ]
                ),

                "zero_padding_verified": (
                    adapter[
                        "zero_padding_verified"
                    ]
                ),

                "zero_padding_nonzero_count": (
                    adapter[
                        "zero_padding_nonzero_count"
                    ]
                ),

                "temporal_strategy": (
                    adapter[
                        "temporal_strategy"
                    ]
                ),

                "note": (
                    adapter["note"]
                ),
            },

            "warning": (
                classification[
                    "warning"
                ]
            ),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "KSL classification failed: "
                f"{exc}"
            ),
        ) from exc

    finally:
        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):
            os.remove(
                temp_path
            )
