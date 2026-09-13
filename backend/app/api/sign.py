import os
import tempfile

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.models.sign import (
    LandmarkExtractionResponse,
)

from app.services.ksl_landmarks import (
    extract_ksl_landmarks,
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