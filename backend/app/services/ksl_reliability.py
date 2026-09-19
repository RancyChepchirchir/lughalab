from typing import Any, Dict, List


MIN_HAND_COVERAGE = 0.20
MIN_TOP_PROBABILITY = 0.55
MIN_PROBABILITY_MARGIN = 0.15


def assess_ksl_reliability(
    *,
    detection_rate: float,
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Diagnostic rejection layer for the closed-set
    LughaLab KSL baseline.

    This is deliberately not presented as a calibrated
    OOD detector. Thresholds are provisional research
    heuristics and must be validated empirically before
    being interpreted as operating points.
    """

    if not predictions:
        raise ValueError(
            "At least one class prediction is required."
        )

    ranked = sorted(
        predictions,
        key=lambda item: item["probability"],
        reverse=True,
    )

    top_probability = float(
        ranked[0]["probability"]
    )

    second_probability = (
        float(ranked[1]["probability"])
        if len(ranked) > 1
        else 0.0
    )

    probability_margin = (
        top_probability
        - second_probability
    )

    checks = {
        "hand_coverage": (
            detection_rate
            >= MIN_HAND_COVERAGE
        ),
        "top_probability": (
            top_probability
            >= MIN_TOP_PROBABILITY
        ),
        "probability_margin": (
            probability_margin
            >= MIN_PROBABILITY_MARGIN
        ),
    }

    reasons = []

    if not checks["hand_coverage"]:
        reasons.append(
            "insufficient_hand_landmark_coverage"
        )

    if not checks["top_probability"]:
        reasons.append(
            "weak_closed_set_softmax_peak"
        )

    if not checks["probability_margin"]:
        reasons.append(
            "ambiguous_closed_set_prediction"
        )

    accepted = all(
        checks.values()
    )

    if accepted:
        decision = "provisionally_accepted"
    else:
        decision = "unreliable"

    return {
        "decision": decision,
        "accepted": accepted,

        "signals": {
            "hand_coverage": float(
                detection_rate
            ),
            "top_probability": (
                top_probability
            ),
            "second_probability": (
                second_probability
            ),
            "probability_margin": (
                probability_margin
            ),
        },

        "checks": checks,

        "thresholds": {
            "minimum_hand_coverage": (
                MIN_HAND_COVERAGE
            ),
            "minimum_top_probability": (
                MIN_TOP_PROBABILITY
            ),
            "minimum_probability_margin": (
                MIN_PROBABILITY_MARGIN
            ),
        },

        "reasons": reasons,

        "note": (
            "Reliability gating currently combines "
            "landmark coverage and closed-set softmax "
            "behaviour. Thresholds are provisional "
            "research heuristics, not calibrated KSL "
            "OOD operating points."
        ),
    }