from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


_DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "train_csv": "data/MP_IN_train.csv",
        "eval_csv": "data/MP_IN_test.csv",
        "concept_csv": "data/df_icd10.csv",
        "output_dir": "outputs/mimic_sapbert",
    },
    "data": {
        "text_col": "text",
        "label_col": "hospital_expire_flag",
    },
    "concepts": {
        "code_col": "code",
        "name_col": "name",
        "group_col": "idx_section",
        "group_offset": 1,
        "use_from_n3c_if_available": True,
        "template": "{name} (Ancestral category: {ancestral_category}, {chapter})",
    },
    "model": {
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "batch_size": 4,
        "max_length": 512,
        "num_workers": 2,
        "concept_batch_size": 4,
        "concept_max_length": 256,
        "concept_pooling": "cls",
        "dv": 256,
        "freeze_text_encoder": False,
        "freeze_concepts": True,
        "attention_activation": "sparsemax",
        "top_k": 4,
        "temperature": 0.07,
        "gate_margin": 0.85,
        "gate_tau": 0.05,
        "null_bias_init": 2.0,
    },
    "training": {
        "seed": 42,
        "deterministic": True,
        "warmup_epochs": 9,
        "concept_epochs": 1,
        "blackbox_lr": 1.0e-6,
        "concept_lr": 5.0e-7,
        "lambda_null_target": 0.02,
        "null_target": 0.95,
        "lambda_entropy": 0.02,
        "lambda_group_lasso": 1.0e-3,
    },
}


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and fill missing values from the notebook-derived defaults."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_update(_DEFAULT_CONFIG, user_cfg)


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False)
