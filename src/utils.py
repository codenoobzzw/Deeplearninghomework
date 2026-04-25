from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np


JSON_INDENT = 2


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(path: os.PathLike[str] | str, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=JSON_INDENT)


def load_json(path: os.PathLike[str] | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_hidden_dims(text_or_value: Any) -> List[int]:
    if isinstance(text_or_value, int):
        return [int(text_or_value)]
    if isinstance(text_or_value, (list, tuple)):
        dims = [int(v) for v in text_or_value]
        if not dims:
            raise ValueError("hidden_dims 不能为空")
        return dims
    if text_or_value is None:
        raise ValueError("hidden_dims 不能为空")
    text = str(text_or_value).strip()
    if not text:
        raise ValueError("hidden_dims 不能为空")
    dims = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not dims:
        raise ValueError("hidden_dims 不能为空")
    return dims


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def chunked(sequence: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    for idx in range(0, len(sequence), chunk_size):
        yield sequence[idx : idx + chunk_size]
