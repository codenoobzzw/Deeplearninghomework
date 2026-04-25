from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .nn import MLPClassifier
from .utils import ensure_dir, load_json, save_json, to_serializable


WEIGHTS_FILE = "weights.npz"
META_FILE = "meta.json"



def save_checkpoint(checkpoint_dir: str | Path, model: MLPClassifier, metadata: dict[str, Any]) -> None:
    checkpoint_dir = ensure_dir(checkpoint_dir)
    weights_path = checkpoint_dir / WEIGHTS_FILE
    meta_path = checkpoint_dir / META_FILE

    np.savez(weights_path, **model.state_dict())
    merged_meta = dict(metadata)
    merged_meta["model"] = model.config()
    save_json(meta_path, to_serializable(merged_meta))



def load_checkpoint(checkpoint_dir: str | Path) -> tuple[MLPClassifier, dict[str, Any]]:
    checkpoint_dir = Path(checkpoint_dir)
    weights_path = checkpoint_dir / WEIGHTS_FILE
    meta_path = checkpoint_dir / META_FILE
    if not weights_path.exists():
        raise FileNotFoundError(f"没有找到权重文件: {weights_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"没有找到元信息文件: {meta_path}")

    metadata = load_json(meta_path)
    model_cfg = metadata["model"]
    model = MLPClassifier(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dims=model_cfg["hidden_dims"],
        num_classes=int(model_cfg["num_classes"]),
        activation=str(model_cfg["activation"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )

    state_npz = np.load(weights_path)
    state_dict = {key: state_npz[key] for key in state_npz.files}
    model.load_state_dict(state_dict)
    return model, metadata
