"""
eda.py — Exploratory Data Analysis module.
Requirement 3: EDA plots, churn rate, correlation matrix, distributions.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


class OutputDirectoryError(Exception):
    pass


@dataclass
class EDAReport:
    churn_rate: float
    risk_distribution: pd.DataFrame
    correlation_matrix: pd.DataFrame
    plots_saved: List[str] = field(default_factory=list)


class EDA_Module:
    """Generates EDA statistics and plots for the CustomerGuard pipeline."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except OSError as e:
                raise OutputDirectoryError(
                    f"Cannot create output directory '{self.output_dir}': {e}"
                )
        if not os.access(self.output_dir, os.W_OK):
            raise OutputDirectoryError(
                f"Output directory '{self.output_dir}' is not writable."
            )

    # ------------------------------------------------------------------
    # Req 3 C1 — Overall churn rate
    # ------------------------------------------------------------------

    def compute_churn_rate(self, df: pd.DataFrame) -> float:
        if "Churn" not in df.columns:
            raise ValueError("Column 'Churn' not found in DataFrame.")
        if len(df) == 0:
            raise ValueError("DataFrame is empty — cannot compute churn rate.")
        rate = float(df["Churn"].sum() / len(df) * 100)
        logger.info("Overall churn rate: %.2f%%", rate)
        return rate

    # ------------------------------------------------------------------
    # Req 3 C2 — Risk-level distribution (post-prediction)
    # ------------------------------------------------------------------

    def compute_risk_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        if "risk_level" not in df.columns:
            return pd.DataFrame(columns=["risk_level", "count", "percentage"])
        counts = df["risk_level"].value_counts().reset_index()
        counts.columns = ["risk_level", "count"]
        counts["percentage"] = (counts["count"] / len(df) * 100).round(2)
        return counts

    # ------------------------------------------------------------------
    # Req 3 C3 — Correlation matrix
    # ------------------------------------------------------------------

    def compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[num_cols].corr()
        # Save heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Matrix")
        self._save_plot(fig, "correlation_matrix.png")
        return corr

    # ------------------------------------------------------------------
    # Req 3 C4 — Distribution histograms per segment
    # ------------------------------------------------------------------

    def plot_histograms(self, df: pd.DataFrame) -> List[str]:
        features = ["tenure", "MonthlyCharges", "TotalCharges"]
        saved = []
        for feat in features:
            if feat not in df.columns:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, (label, churn_val) in zip(axes, [("Churned", 1), ("Not Churned", 0)]):
                segment = df[df["Churn"] == churn_val][feat].dropna()
                ax.hist(segment, bins=30, color="steelblue" if churn_val == 0 else "tomato",
                        edgecolor="white", alpha=0.85)
                ax.set_title(f"{feat} — {label}")
                ax.set_xlabel(feat)
                ax.set_ylabel("Count")
            fig.tight_layout()
            fname = f"histogram_{feat}.png"
            self._save_plot(fig, fname)
            saved.append(fname)
        return saved

    # ------------------------------------------------------------------
    # Req 3 C5 — Churn rate by categorical feature
    # ------------------------------------------------------------------

    def plot_churn_rate_by_category(self, df: pd.DataFrame) -> List[str]:
        features = ["Contract", "PaymentMethod", "InternetService"]
        saved = []
        for feat in features:
            if feat not in df.columns:
                continue
            unique_vals = df[feat].dropna().unique()
            if len(unique_vals) < 2 or len(unique_vals) > 20:
                logger.warning(
                    "Skipping churn-rate bar chart for '%s': %d unique values (expected 2–20).",
                    feat, len(unique_vals)
                )
                continue
            churn_rates = (
                df.groupby(feat)["Churn"]
                .apply(lambda x: x.mean() * 100)
                .reset_index()
            )
            churn_rates.columns = [feat, "churn_rate"]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(churn_rates[feat].astype(str), churn_rates["churn_rate"],
                   color="steelblue", edgecolor="white")
            ax.set_title(f"Churn Rate by {feat}")
            ax.set_xlabel(feat)
            ax.set_ylabel("Churn Rate (%)")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            fname = f"bar_churn_rate_{feat}.png"
            self._save_plot(fig, fname)
            saved.append(fname)
        return saved

    # ------------------------------------------------------------------
    # Run all EDA
    # ------------------------------------------------------------------

    def run_all(self, df: pd.DataFrame) -> EDAReport:
        logger.info("Running EDA...")
        churn_rate = self.compute_churn_rate(df)
        risk_dist = self.compute_risk_distribution(df)
        corr = self.compute_correlation_matrix(df)
        hist_files = self.plot_histograms(df)
        bar_files = self.plot_churn_rate_by_category(df)
        all_plots = hist_files + bar_files + ["correlation_matrix.png"]
        logger.info("EDA complete. %d plots saved to '%s'", len(all_plots), self.output_dir)
        return EDAReport(
            churn_rate=churn_rate,
            risk_distribution=risk_dist,
            correlation_matrix=corr,
            plots_saved=all_plots,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _save_plot(self, fig, filename: str):
        self._ensure_output_dir()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved: '%s'", path)
