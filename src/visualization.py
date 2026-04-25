from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import ImageRecord, ImageRepository
from .utils import ensure_dir



def _prepare_path(path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    return path



def plot_loss_curves(history: dict, output_path: str | Path) -> None:
    output_path = _prepare_path(output_path)
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("train_loss", []), label="train_loss")
    plt.plot(epochs, history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()



def plot_accuracy_curves(history: dict, output_path: str | Path) -> None:
    output_path = _prepare_path(output_path)
    epochs = np.arange(1, len(history.get("val_accuracy", [])) + 1)
    plt.figure(figsize=(8, 5))
    if "train_accuracy" in history:
        plt.plot(epochs, history.get("train_accuracy", []), label="train_accuracy")
    plt.plot(epochs, history.get("val_accuracy", []), label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()



def plot_lr_curve(history: dict, output_path: str | Path) -> None:
    output_path = _prepare_path(output_path)
    epochs = np.arange(1, len(history.get("learning_rate", [])) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("learning_rate", []), label="learning_rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()



def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], output_path: str | Path) -> None:
    output_path = _prepare_path(output_path)
    cm = np.asarray(cm)
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()



def save_weight_grid(
    first_layer_weight: np.ndarray,
    input_shape: Sequence[int],
    output_path: str | Path,
    max_filters: int = 64,
    cols: int = 8,
) -> None:
    output_path = _prepare_path(output_path)
    input_shape = tuple(int(v) for v in input_shape)
    if len(input_shape) != 3:
        raise ValueError(f"期望输入形状为 (H, W, C)，实际得到: {input_shape}")

    filters = first_layer_weight.T
    num_filters = min(int(max_filters), filters.shape[0])
    filters = filters[:num_filters]
    cols = max(1, int(cols))
    rows = int(math.ceil(num_filters / cols))

    plt.figure(figsize=(cols * 2, rows * 2))
    for idx in range(num_filters):
        plt.subplot(rows, cols, idx + 1)
        weight_img = filters[idx].reshape(input_shape)
        min_v, max_v = float(weight_img.min()), float(weight_img.max())
        if max_v - min_v < 1e-8:
            vis = np.zeros_like(weight_img)
        else:
            vis = (weight_img - min_v) / (max_v - min_v)
        if input_shape[2] == 1:
            plt.imshow(vis.squeeze(-1), cmap="gray")
        else:
            plt.imshow(vis)
        plt.axis("off")
        plt.title(f"h{idx}", fontsize=9)

    plt.suptitle("First Hidden Layer Weights", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()



def save_misclassified_grid(
    records: Sequence[ImageRecord],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    repository: ImageRepository,
    class_names: Sequence[str],
    output_path: str | Path,
    max_examples: int = 16,
    cols: int = 4,
) -> int:
    output_path = _prepare_path(output_path)
    mistakes = [idx for idx, (t, p) in enumerate(zip(y_true, y_pred)) if int(t) != int(p)]
    mistakes = mistakes[: max(1, int(max_examples))]
    if not mistakes:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No misclassified examples on this split.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close()
        return 0

    cols = max(1, int(cols))
    rows = int(math.ceil(len(mistakes) / cols))
    plt.figure(figsize=(cols * 3.2, rows * 3.2))
    for plot_idx, data_idx in enumerate(mistakes):
        plt.subplot(rows, cols, plot_idx + 1)
        image = repository.load_image(records[data_idx])
        plt.imshow(image)
        true_name = class_names[int(y_true[data_idx])]
        pred_name = class_names[int(y_pred[data_idx])]
        plt.title(f"T:{true_name}\nP:{pred_name}", fontsize=9)
        plt.axis("off")

    plt.suptitle("Misclassified Test Examples", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return len(mistakes)
