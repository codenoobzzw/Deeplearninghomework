from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np

from .utils import load_json, to_serializable



def load_search_config(path: str | Path) -> dict[str, Any]:
    return load_json(path)



def _grid_trials(param_space: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(param_space.keys())
    values: list[list[Any]] = []
    for key in keys:
        value = param_space[key]
        if not isinstance(value, list):
            value = [value]
        values.append(value)
    trials = []
    for combo in itertools.product(*values):
        trial = {key: value for key, value in zip(keys, combo)}
        trials.append(trial)
    return trials



def _sample_from_spec(spec: Any, rng: np.random.Generator) -> Any:
    if isinstance(spec, list):
        return spec[int(rng.integers(0, len(spec)))]
    if not isinstance(spec, dict):
        return spec

    spec_type = spec.get("type", "choice")
    if spec_type == "choice":
        choices = spec["values"]
        return choices[int(rng.integers(0, len(choices)))]
    if spec_type == "uniform":
        low = float(spec["low"])
        high = float(spec["high"])
        return float(rng.uniform(low, high))
    if spec_type == "log_uniform":
        low = float(spec["low"])
        high = float(spec["high"])
        if low <= 0 or high <= 0:
            raise ValueError("log_uniform 的 low/high 必须大于 0")
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    if spec_type == "int_uniform":
        low = int(spec["low"])
        high = int(spec["high"])
        return int(rng.integers(low, high + 1))
    raise ValueError(f"不支持的随机搜索类型: {spec_type}")



def _random_trials(param_space: dict[str, Any], num_trials: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    trials: list[dict[str, Any]] = []
    for _ in range(num_trials):
        trial = {key: _sample_from_spec(spec, rng) for key, spec in param_space.items()}
        trials.append(trial)
    return trials



def generate_trials(search_config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    search_type = search_config.get("search_type", "grid").lower()
    param_space = search_config.get("param_space", {})
    if not param_space:
        raise ValueError("search_config 中缺少 param_space")

    if search_type == "grid":
        return _grid_trials(param_space)
    if search_type == "random":
        num_trials = int(search_config.get("num_trials", 10))
        return _random_trials(param_space, num_trials=num_trials, seed=seed)
    raise ValueError(f"不支持的搜索方式: {search_type}")



def merge_trial_config(base_config: dict[str, Any], trial_config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(base_config)
    merged.update(trial_config)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged



def sanitize_trial_name(trial_idx: int, config: dict[str, Any]) -> str:
    def _fmt(value: Any) -> str:
        if isinstance(value, list):
            return "x".join(str(v) for v in value)
        if isinstance(value, float):
            return f"{value:.3g}"
        return str(value)

    parts = [f"trial_{trial_idx:03d}"]
    for key in ["learning_rate", "hidden_dims", "activation", "weight_decay"]:
        if key not in config:
            continue
        value_str = _fmt(config[key]).replace("/", "-").replace(" ", "")
        parts.append(f"{key}-{value_str}")
    return "__".join(parts)
