"""
shap_explainer.py — SHAP-based prediction explanation.
Requirement 7: Compute SHAP values, top-N features, force plots, summary plots.
"""

import os
import logging
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class SHAPError(Exception):
    pass


@dataclass
class FeatureContribution:
    feature_name: str
    shap_value: float
    direction: str  # "increases_churn" | "decreases_churn"


@dataclass
class SHAPResult:
    customer_id: str
    shap_values: np.ndarray       # shape: (n_features,)
    feature_names: List[str]
    base_value: float
    top_features: List[FeatureContribution] = field(default_factory=list)


class SHAP_Explainer:
    """
    Selects TreeExplainer or LinearExplainer based on model type,
    then computes per-customer and global SHAP explanations.
    """

    def __init__(self, model, feature_names: List[str],
                 top_n: int = 5, output_dir: str = "results/shap/"):
        if top_n < 1:
            raise SHAPError(f"top_n must be >= 1, got {top_n}.")
        if top_n > len(feature_names):
            raise SHAPError(
                f"top_n ({top_n}) cannot exceed number of features ({len(feature_names)})."
            )
        self._model = model
        self.feature_names = feature_names
        self.top_n = top_n
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._explainer = self._build_explainer()

    # ------------------------------------------------------------------
    # Req 7 C6–C8 — Backend selection
    # ------------------------------------------------------------------

    def _build_explainer(self):
        if isinstance(self._model, LogisticRegression):
            logger.info("SHAP: using LinearExplainer for LogisticRegression")
            return shap.LinearExplainer(self._model, shap.maskers.Independent(
                np.zeros((1, len(self.feature_names)))
            ))
        elif isinstance(self._model, (RandomForestClassifier, XGBClassifier)):
            logger.info("SHAP: using TreeExplainer for %s",
                        type(self._model).__name__)
            return shap.TreeExplainer(self._model)
        else:
            logger.warning("Unknown model type — using KernelExplainer (slow).")
            return shap.KernelExplainer(
                self._model.predict_proba,
                np.zeros((1, len(self.feature_names))),
            )

    # ------------------------------------------------------------------
    # Req 7 C1 — Explain single customer
    # ------------------------------------------------------------------

    def explain_single(self, feature_vector: np.ndarray,
                       customer_id: str = "unknown") -> SHAPResult:
        fv = feature_vector.reshape(1, -1)
        raw = self._explainer.shap_values(fv)

        # For binary classifiers shap_values may return a list [class0, class1]
        if isinstance(raw, list):
            sv = raw[1][0]          # class 1 = churn
        else:
            sv = raw[0]

        base = self._base_value()
        result = SHAPResult(
            customer_id=customer_id,
            shap_values=sv,
            feature_names=list(self.feature_names),
            base_value=base,
        )
        result.top_features = self.top_features(result)
        return result

    # ------------------------------------------------------------------
    # Req 7 C2–C3 — Top-N feature contributions
    # ------------------------------------------------------------------

    def top_features(self, result: SHAPResult) -> List[FeatureContribution]:
        indices = np.argsort(np.abs(result.shap_values))[::-1][: self.top_n]
        contributions = []
        for idx in indices:
            sv = result.shap_values[idx]
            contributions.append(FeatureContribution(
                feature_name=result.feature_names[idx],
                shap_value=float(sv),
                direction="increases_churn" if sv > 0 else "decreases_churn",
            ))
        return contributions

    # ------------------------------------------------------------------
    # Req 7 C4 — SHAP summary bar plot (global, test set)
    # ------------------------------------------------------------------

    def plot_summary(self, X_test: np.ndarray) -> None:
        raw = self._explainer.shap_values(X_test)
        if isinstance(raw, list):
            sv = raw[1]
        else:
            sv = raw

        fig, ax = plt.subplots(figsize=(9, 6))
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:20]
        ax.barh(
            [self.feature_names[i] for i in order][::-1],
            mean_abs[order][::-1],
            color="steelblue",
        )
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Global Feature Importance (SHAP)")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "shap_summary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP summary plot saved: '%s'", path)

    # ------------------------------------------------------------------
    # Req 7 C5 — SHAP force plot per customer (HTML)
    # ------------------------------------------------------------------

    def plot_force(self, result: SHAPResult) -> None:
        base = self._base_value()
        force = shap.force_plot(
            base,
            result.shap_values,
            result.feature_names,
            matplotlib=False,
        )
        filename = f"force_{result.customer_id}.html"
        path = os.path.join(self.output_dir, filename)
        shap.save_html(path, force)
        logger.info("Force plot saved: '%s'", path)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _base_value(self) -> float:
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[1])
        return float(ev)
