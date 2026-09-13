from typing import List, Optional

from pydantic import BaseModel, Field


class BenchmarkRequest(BaseModel):
    text_a: str = Field(..., min_length=1, max_length=2000)
    text_b: str = Field(..., min_length=1, max_length=2000)


class ModelBenchmarkResult(BaseModel):
    model: str
    device: str
    embedding_dimension: int
    similarity: float
    inference_time_ms: float


class BenchmarkResponse(BaseModel):
    text_a: str
    text_b: str
    language: str
    results: List[ModelBenchmarkResult]
    fastest_model: Optional[str] = None
    highest_similarity_model: Optional[str] = None
    note: Optional[str] = None