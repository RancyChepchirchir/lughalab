from fastapi import FastAPI

from app.api.nlp import (
    router as nlp_router,
)

from app.api.sign import (
    router as sign_router,
)

from app.api.speech import (
    router as speech_router,
)


app = FastAPI(
    title="LughaLab API",
    description=(
        "African Language & Sign Intelligence — "
        "a research API for low-resource NLP, "
        "speech and sign-language modelling."
    ),
    version="0.3.0",
)


@app.get("/")
def root():
    return {
        "name": "LughaLab",
        "subtitle": (
            "African Language "
            "& Sign Intelligence"
        ),
        "status": (
            "research-preview"
        ),
        "version": "0.3.0",
        "modalities": [
            "text",
            "speech",
            "sign",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


app.include_router(
    nlp_router
)

app.include_router(
    speech_router
)

app.include_router(
    sign_router
)