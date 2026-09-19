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

class KSLClassProbability(BaseModel):
    index: int
    label: str
    probability: float


class KSLAdapterDiagnostics(BaseModel):
    native_shape: List[int]
    hands_shape: List[int]
    checkpoint_shape: List[int]
    zero_padding_verified: bool
    zero_padding_nonzero_count: int
    temporal_strategy: str
    note: Optional[str] = None

class KSLReliabilitySignals(BaseModel):
    hand_coverage: float
    top_probability: float
    second_probability: float
    probability_margin: float


class KSLReliabilityChecks(BaseModel):
    hand_coverage: bool
    top_probability: bool
    probability_margin: bool


class KSLReliabilityThresholds(BaseModel):
    minimum_hand_coverage: float
    minimum_top_probability: float
    minimum_probability_margin: float


class KSLReliabilityAssessment(BaseModel):
    decision: str
    accepted: bool

    signals: KSLReliabilitySignals
    checks: KSLReliabilityChecks
    thresholds: KSLReliabilityThresholds

    reasons: List[str]

    note: Optional[str] = None

class KSLOpenSetAssessment(BaseModel):
    is_ood: bool
    open_set_label: str
    ood_score: float
    ood_threshold: float
    ood_detector: str
    nearest_known_label: str
    nearest_labels: List[str]
    nearest_distances: List[float]
    calibration: str

class KSLClassificationResponse(BaseModel):
    filename: str

    predicted_label: str
    predicted_probability: float

    predictions: List[
        KSLClassProbability
    ]
    reliability: KSLReliabilityAssessment
    open_set: KSLOpenSetAssessment

    model: str
    device: str

    closed_set: bool
    supported_labels: List[str]

    frames_read: int
    frames_detected: int
    detection_rate: float

    adapter: KSLAdapterDiagnostics

    warning: str