"""
config_loader.py — Reads and validates config.yaml for CustomerGuard.
Raises ConfigError for any missing required parameter.
"""

import yaml
from dataclasses import dataclass, field
from typing import List


class ConfigError(Exception):
    """Raised when a required configuration parameter is absent."""
    pass


# ---------------------------------------------------------------------------
# Dataclasses mirroring config.yaml structure
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    dataset_path: str
    output_dir: str
    log_file: str
    random_seed: int
    stages: List[str]


@dataclass
class PreprocessorConfig:
    artifact_path: str


@dataclass
class ClassifierConfig:
    test_split_ratio: float
    artifact_path: str
    output_dir: str


@dataclass
class SHAPConfig:
    top_n: int
    output_dir: str


@dataclass
class RiskConfig:
    low_max_threshold: float
    medium_max_threshold: float


@dataclass
class DashboardConfig:
    results_path: str
    port: int


@dataclass
class AppConfig:
    pipeline: PipelineConfig
    preprocessor: PreprocessorConfig
    classifier: ClassifierConfig
    shap: SHAPConfig
    risk: RiskConfig
    dashboard: DashboardConfig


# ---------------------------------------------------------------------------
# Required keys — absence halts the pipeline with a named error
# ---------------------------------------------------------------------------

_REQUIRED = {
    "pipeline.dataset_path",
    "pipeline.output_dir",
    "classifier.test_split_ratio",
    "classifier.artifact_path",
    "shap.top_n",
    "risk.low_max_threshold",
    "risk.medium_max_threshold",
}


def _get_nested(d: dict, dotted_key: str):
    """Return the value at a dotted key path, or raise KeyError."""
    keys = dotted_key.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(dotted_key)
        cur = cur[k]
    return cur


def load_config(path: str = "config.yaml") -> AppConfig:
    """
    Load and validate config.yaml.

    Raises
    ------
    ConfigError
        If the file cannot be read or a required parameter is missing.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found: '{path}'")
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse configuration file '{path}': {exc}")

    # Validate all required keys
    missing = []
    for key in sorted(_REQUIRED):
        try:
            _get_nested(raw, key)
        except KeyError:
            missing.append(key)

    if missing:
        raise ConfigError(
            "The following required configuration parameters are absent: "
            + ", ".join(missing)
        )

    p = raw.get("pipeline", {})
    pp = raw.get("preprocessor", {})
    cl = raw.get("classifier", {})
    sh = raw.get("shap", {})
    rk = raw.get("risk", {})
    db = raw.get("dashboard", {})

    return AppConfig(
        pipeline=PipelineConfig(
            dataset_path=p["dataset_path"],
            output_dir=p["output_dir"],
            log_file=p.get("log_file", p["output_dir"] + "pipeline.log"),
            random_seed=p.get("random_seed", 42),
            stages=p.get(
                "stages",
                [
                    "data_ingestion", "preprocessing", "eda", "model_training",
                    "prediction", "risk_classification", "shap_explanation",
                    "recommendation", "dashboard",
                ],
            ),
        ),
        preprocessor=PreprocessorConfig(
            artifact_path=pp.get("artifact_path", "artifacts/preprocessor.pkl"),
        ),
        classifier=ClassifierConfig(
            test_split_ratio=cl["test_split_ratio"],
            artifact_path=cl["artifact_path"],
            output_dir=cl.get("output_dir", p["output_dir"]),
        ),
        shap=SHAPConfig(
            top_n=sh["top_n"],
            output_dir=sh.get("output_dir", p["output_dir"] + "shap/"),
        ),
        risk=RiskConfig(
            low_max_threshold=rk["low_max_threshold"],
            medium_max_threshold=rk["medium_max_threshold"],
        ),
        dashboard=DashboardConfig(
            results_path=db.get("results_path", p["output_dir"] + "predictions.csv"),
            port=db.get("port", 8501),
        ),
    )
