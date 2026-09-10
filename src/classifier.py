"""
classifier.py — Train, evaluate, and serialize ML models.
Requirement 4: Logistic Regression, Random Forest, XGBoost.
"""

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    mean_inference_ms: float


class Classifier:
    """
    Trains Logistic Regression, Random Forest, and XGBoost on churn data,
    evaluates each model, and serializes the best (highest AUC-ROC).
    """

    def __init__(self, test_split_ratio: float = 0.20,
                 random_seed: int = 42, output_dir: str = "results/plots/"):
        self.test_split_ratio = test_split_ratio
        self.random_seed = random_seed
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=random_seed, solver="lbfgs"
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200, random_state=random_seed, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200, random_state=random_seed,
                use_label_encoder=False, eval_metric="logloss",
                verbosity=0,
            ),
        }
        self._trained_models = {}
        self._best_model = None
        self._best_model_name = None
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Req 4 C1 — Stratified split
    # ------------------------------------------------------------------

    def split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
        """
        Split df into (X_train, X_test, y_train, y_test) using stratified sampling.
        """
        if "Churn" not in df.columns:
            raise ValueError("Target column 'Churn' not found.")

        target = df["Churn"].values
        features = df.drop(columns=["Churn", "customerID"], errors="ignore")
        self._feature_names = features.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            features.values, target,
            test_size=self.test_split_ratio,
            stratify=target,
            random_state=self.random_seed,
        )
        logger.info(
            "Split: %d train / %d test (stratified, seed=%d)",
            len(X_train), len(X_test), self.random_seed,
        )
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Req 4 C2 — Train all models
    # ------------------------------------------------------------------

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        for name, model in self._models.items():
            logger.info("Training %s ...", name)
            t0 = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - t0
            self._trained_models[name] = model
            logger.info("  %s trained in %.2f seconds.", name, elapsed)

    # ------------------------------------------------------------------
    # Req 4 C3 — Evaluate all models
    # ------------------------------------------------------------------

    def evaluate_all(self, X_test: np.ndarray,
                     y_test: np.ndarray) -> List[ModelMetrics]:
        metrics_list = []
        for name, model in self._trained_models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Measure mean inference latency over 100 single-record runs
            sample = X_test[0:1]
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                model.predict_proba(sample)
                times.append((time.perf_counter() - t0) * 1000)
            mean_ms = float(np.mean(times))

            m = ModelMetrics(
                name=name,
                accuracy=float(accuracy_score(y_test, y_pred)),
                precision=float(precision_score(y_test, y_pred, average="binary")),
                recall=float(recall_score(y_test, y_pred, average="binary")),
                f1=float(f1_score(y_test, y_pred, average="binary")),
                auc_roc=float(roc_auc_score(y_test, y_prob)),
                mean_inference_ms=mean_ms,
            )
            metrics_list.append(m)
            logger.info(
                "%s — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | "
                "AUC-ROC: %.4f | Latency: %.2f ms",
                name, m.accuracy, m.precision, m.recall,
                m.f1, m.auc_roc, m.mean_inference_ms,
            )
        return metrics_list

    # ------------------------------------------------------------------
    # Req 4 C6, C7 — Select best model
    # ------------------------------------------------------------------

    def select_best(self, metrics: List[ModelMetrics]):
        best = max(
            metrics,
            key=lambda m: (m.auc_roc, -m.mean_inference_ms),
        )
        self._best_model_name = best.name
        self._best_model = self._trained_models[best.name]
        logger.info("Best model: %s (AUC-ROC=%.4f, latency=%.2f ms)",
                    best.name, best.auc_roc, best.mean_inference_ms)
        return self._best_model

    # ------------------------------------------------------------------
    # Req 4 C6 — Serialize best model
    # ------------------------------------------------------------------

    def save_best(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "model": self._best_model,
            "model_name": self._best_model_name,
            "feature_names": self._feature_names,
        }, path)
        logger.info("Best model (%s) saved to '%s'", self._best_model_name, path)

    # ------------------------------------------------------------------
    # Req 4 C4 — Confusion matrices
    # ------------------------------------------------------------------

    def plot_confusion_matrices(self, X_test: np.ndarray,
                                y_test: np.ndarray) -> None:
        for name, model in self._trained_models.items():
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns_labels = ["No Churn", "Churn"]
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(2), yticks=np.arange(2),
                xticklabels=sns_labels, yticklabels=sns_labels,
                xlabel="Predicted", ylabel="Actual",
                title=f"{name} — Confusion Matrix",
            )
            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(cm[i, j], "d"),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            path = os.path.join(self.output_dir, f"{name}_confusion_matrix.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Confusion matrix saved: '%s'", path)

    # ------------------------------------------------------------------
    # Req 4 C5 — ROC curves
    # ------------------------------------------------------------------

    def plot_roc_curves(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["steelblue", "tomato", "seagreen"]
        for (name, model), color in zip(self._trained_models.items(), colors):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="ROC Curves — All Models")
        ax.legend(loc="lower right")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "roc_curves_all_models.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("ROC curves saved: '%s'", path)

    def get_feature_names(self) -> List[str]:
        return list(self._feature_names)

    def get_best_model(self):
        return self._best_model

    def get_best_model_name(self) -> str:
        return self._best_model_name
