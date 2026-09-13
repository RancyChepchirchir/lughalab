import os
import tempfile

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.models.speech import (
    TranscriptionResponse,
)

from app.services.swahili_asr import (
    transcribe_swahili,
)


router = APIRouter(
    prefix="/speech",
    tags=["speech"],
)


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
)
async def transcribe(
    file: UploadFile = File(...),
):
    suffix = os.path.splitext(
        file.filename or ""
    )[1]

    if not suffix:
        suffix = ".wav"

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

        return transcribe_swahili(
            temp_path
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Swahili transcription failed: "
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