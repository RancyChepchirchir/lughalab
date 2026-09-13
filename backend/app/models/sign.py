from typing import List, Optional

from pydantic import BaseModel


class LandmarkFrame(BaseModel):
    frame_index: int
    timestamp_seconds: float
    landmarks: List[float]


class LandmarkExtractionResponse(BaseModel):
    filename: str
    frames_read: int
    frames_detected: int
    feature_dimension: int
    duration_seconds: float
    detection_rate: float
    frames: List[LandmarkFrame]
    note: Optional[str] = None