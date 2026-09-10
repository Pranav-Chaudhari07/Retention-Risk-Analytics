"""
predictor.py — Loads a model artifact and performs churn probability inference.
Requirement 5: Single and batch churn probability prediction.
"""

import os
import time
import logging
import joblib
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


class PredictorError(Exception):
    pass


class Predictor:
    """
    Loads a serialized model artifact and predicts churn probabilities.
    Supports single-record and batch inference.
    """

    def __init__(self, artifact_path: str):
        if not os.path.exists(artifact_path):
            raise PredictorError(
                f"Model artifact not found at '{artifact_path}'. "
                "Run model_training stage first."
            )
        data = joblib.load(artifact_path)
        self._model = data["model"]
        self._model_name = data.get("model_name", "unknown")
        self._feature_names = data.get("feature_names", [])
        self._expected_dim = len(self._feature_names)
        logger.info(
            "Predictor loaded model '%s' from '%s' (expected dim=%d)",
            self._model_name, artifact_path, self._expected_dim,
        )

    # ------------------------------------------------------------------
    # Req 5 C1 — Single-record prediction (≤ 500 ms)
    # ------------------------------------------------------------------

    def predict_single(self, feature_vector: np.ndarray) -> float:
        self.validate_vector(feature_vector, position=0)
        t0 = time.perf_counter()
        prob = float(self._model.predict_proba(feature_vector.reshape(1, -1))[0, 1])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 500:
            logger.warning("Single-record inference took %.1f ms (SLA: 500 ms)", elapsed_ms)
        return prob

    # ------------------------------------------------------------------
    # Req 5 C2 — Batch prediction (≤ 60 s for 10k records)
    # ------------------------------------------------------------------

    def predict_batch(self, feature_vectors: List[np.ndarray]) -> List[float]:
        if not feature_vectors:
            return []
        if len(feature_vectors) > 10_000:
            raise PredictorError(
                f"Batch size {len(feature_vectors)} exceeds the maximum of 10,000."
            )

        # Validate all records first — reject entire batch on any error
        errors = []
        for i, fv in enumerate(feature_vectors):
            try:
                self.validate_vector(fv, position=i)
            except PredictorError as exc:
                errors.append(str(exc))
        if errors:
            raise PredictorError(
                f"Batch rejected — {len(errors)} invalid record(s):\n" +
                "\n".join(errors)
            )

        t0 = time.perf_counter()
        X = np.stack(feature_vectors)
        probs = self._model.predict_proba(X)[:, 1].tolist()
        elapsed_s = time.perf_counter() - t0
        if elapsed_s > 60:
            logger.warning(
                "Batch inference for %d records took %.1f s (SLA: 60 s)",
                len(feature_vectors), elapsed_s,
            )
        return [float(p) for p in probs]

    # ------------------------------------------------------------------
    # Req 5 C3, C5 — Validation
    # ------------------------------------------------------------------

    def validate_vector(self, fv: np.ndarray, position: int) -> None:
        """
        Validate dimensionality and content of a feature vector.

        Raises PredictorError with a descriptive message on failure.
        """
        if not isinstance(fv, np.ndarray):
            raise PredictorError(
                f"Record at position {position}: expected np.ndarray, "
                f"got {type(fv).__name__}."
            )
        if fv.ndim != 1:
            raise PredictorError(
                f"Record at position {position}: expected 1-D array, "
                f"got shape {fv.shape}."
            )
        if self._expected_dim > 0 and len(fv) != self._expected_dim:
            raise PredictorError(
                f"Record at position {position}: dimensionality mismatch — "
                f"expected {self._expected_dim}, received {len(fv)}."
            )
        if np.any(np.isnan(fv)) or np.any(np.isinf(fv)):
            bad_indices = np.where(~np.isfinite(fv))[0].tolist()
            raise PredictorError(
                f"Record at position {position}: contains NaN/Inf values "
                f"at feature indices {bad_indices}."
            )
