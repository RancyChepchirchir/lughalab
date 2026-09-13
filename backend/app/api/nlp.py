from fastapi import APIRouter, HTTPException

from app.models.nlp import (
    AnalyseRequest,
    AnalyseResponse,
    POSRequest,
    POSResponse,
)

from app.services.swahili_encoder import (
    analyse_swahili,
)

from app.services.swahili_pos import (
    tag_swahili_pos,
)

from app.models.benchmark import (
    BenchmarkRequest,
    BenchmarkResponse,
)

from app.services.model_benchmark import (
    benchmark_models,
)

router = APIRouter(
    prefix="/nlp",
    tags=["Swahili NLP"],
)


# ============================================================
# SwahBERT representation
# ============================================================


@router.post(
    "/analyse",
    response_model=AnalyseResponse,
)
def analyse(
    request: AnalyseRequest,
):
    try:
        return analyse_swahili(
            request.text
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
                "Swahili model inference failed: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# POS tagging
# ============================================================


@router.post(
    "/pos",
    response_model=POSResponse,
)
def pos_tag(
    request: POSRequest,
):
    try:
        return tag_swahili_pos(
            request.text
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
                "Swahili POS inference failed: "
                f"{exc}"
            ),
        ) from exc

@router.post(
    "/benchmark",
    response_model=BenchmarkResponse,
)
def benchmark(
    request: BenchmarkRequest,
):
    try:
        return benchmark_models(
            request.text_a,
            request.text_b,
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
                "Model benchmark failed: "
                f"{exc}"
            ),
        ) from exc