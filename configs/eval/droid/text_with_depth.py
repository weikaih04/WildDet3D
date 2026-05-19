"""DROID eval - text prompt, GT depth (FoundationStereo)."""
from __future__ import annotations

from vis4d.config.typing import ExperimentConfig

from configs.eval.droid._base import build_droid_eval_config


def get_config() -> ExperimentConfig:
    return build_droid_eval_config(
        exp_name="wilddet3d_droid_text_with_depth",
        oracle_eval=False,
        with_depth=True,
    )
