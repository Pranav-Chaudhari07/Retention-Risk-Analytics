"""
risk_classifier.py — Maps churn probability to Low / Medium / High risk level.
Requirement 6: Risk classification with configurable thresholds.
"""

from typing import List, Union

# Default thresholds (overridden by config.yaml)
DEFAULT_LOW_MAX = 0.40
DEFAULT_MEDIUM_MAX = 0.70


class RiskClassificationError(Exception):
    pass


def classify_risk(
    probability: float,
    low_max: float = DEFAULT_LOW_MAX,
    medium_max: float = DEFAULT_MEDIUM_MAX,
) -> str:
    """
    Map a churn probability to a Risk_Level string.

    Parameters
    ----------
    probability : float
        Churn probability in [0.0, 1.0].
    low_max : float
        Upper boundary (exclusive) for Low risk.
    medium_max : float
        Upper boundary (exclusive) for Medium risk.

    Returns
    -------
    str : "Low" | "Medium" | "High"

    Raises
    ------
    RiskClassificationError
        If probability is outside [0.0, 1.0].
    """
    if not isinstance(probability, (int, float)):
        raise RiskClassificationError(
            f"Expected a numeric probability, got {type(probability).__name__}."
        )
    if probability < 0.0 or probability > 1.0:
        raise RiskClassificationError(
            f"Churn probability {probability} is outside the valid range [0.0, 1.0]."
        )
    if probability < low_max:
        return "Low"
    if probability < medium_max:
        return "Medium"
    return "High"


def classify_risk_batch(
    probabilities: List[float],
    low_max: float = DEFAULT_LOW_MAX,
    medium_max: float = DEFAULT_MEDIUM_MAX,
) -> List[str]:
    """
    Classify a list of churn probabilities.

    Each element is classified independently.  If any element is invalid,
    the full list of error messages is raised without partial results.

    Returns
    -------
    list[str] : ordered list of Risk_Level values.
    """
    errors = []
    for i, p in enumerate(probabilities):
        try:
            classify_risk(p, low_max, medium_max)
        except RiskClassificationError as exc:
            errors.append(f"  Position {i}: {exc}")

    if errors:
        raise RiskClassificationError(
            f"Batch rejected — {len(errors)} invalid value(s):\n" +
            "\n".join(errors)
        )

    return [classify_risk(p, low_max, medium_max) for p in probabilities]
