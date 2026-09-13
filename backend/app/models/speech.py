from typing import List, Optional

from pydantic import BaseModel


class SpeechSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    model: str
    language: str
    device: str
    sampling_rate: int
    duration_seconds: float
    transcription: str
    inference_time_ms: float
    segments: List[SpeechSegment] = []
    note: Optional[str] = None