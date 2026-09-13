from typing import List, Optional

from pydantic import BaseModel, Field


class AnalyseRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Swahili text to analyse.",
    )


class TokenRepresentation(BaseModel):
    token: str
    token_id: int


class AnalyseResponse(BaseModel):
    text: str
    model: str
    language: str
    device: str

    tokens: List[TokenRepresentation]

    token_count: int
    embedding_dimension: int

    sentence_embedding: List[float]

    embedding_norm: float
    inference_time_ms: float

    note: Optional[str] = None


# ============================================================
# POS tagging
# ============================================================


class POSRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Swahili text to POS tag.",
    )


class POSToken(BaseModel):
    token: str
    tag: str
    confidence: float


class POSResponse(BaseModel):
    text: str
    language: str
    model: str
    device: str

    tokens: List[POSToken]

    token_count: int
    inference_time_ms: float

    note: Optional[str] = None