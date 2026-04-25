from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .autograd import Tensor, cross_entropy, l2_regularization
from .checkpoint import save_checkpoint
from .data import ImageRecord, ImageRepository, batch_iterator
from .metrics import accuracy_score, cross_entropy_from_logits
from .optim import ExponentialLRScheduler, SGD
from .utils import ensure_dir, format_seconds, save_json, to_serializable
from .visualization import plot_accuracy_curves, plot_loss_curves, plot_lr_curve


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.01
    momentum: float = 0.9
    lr_decay: float = 1.0
    weight_decay: float = 0.0005
    max_grad_norm: float = 5.0
    seed: int = 42
    use_train_augmentation: bool = True
    train_hflip_prob: float = 0.5
    train_vflip_prob: float = 0.5
    train_rotate_prob: float = 0.5
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4
    include_bias_in_weight_decay: bool = False



def clip_grad_norm(parameters: Sequence[Tensor], max_grad_norm: float) -> float:
    if max_grad_norm <= 0:
        return 0.0

    total_norm_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total_norm_sq += float(np.sum(param.grad**2))

    total_norm = float(np.sqrt(total_norm_sq))
    if total_norm > max_grad_norm:
        clip_coef = float(max_grad_norm / (total_norm + 1e-6))
        for param in parameters:
            if param.grad is not None:
                param.grad *= clip_coef
    return total_norm


def evaluate_split(
    model,
    repository: ImageRepository,
    records: Sequence[ImageRecord],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch_records in batch_iterator(records, batch_size=batch_size, shuffle=False):
        x_batch, y_batch = repository.records_to_batch(batch_records, mean=mean, std=std, flatten=True)
        logits = model.forward_numpy(x_batch)
        batch_loss = cross_entropy_from_logits(logits, y_batch)
        predictions = logits.argmax(axis=1)

        total_loss += batch_loss * x_batch.shape[0]
        total_correct += int((predictions == y_batch).sum())
        total_samples += int(x_batch.shape[0])
        all_predictions.append(predictions)
        all_targets.append(y_batch)

    if total_samples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "predictions": np.asarray([], dtype=np.int64),
            "targets": np.asarray([], dtype=np.int64),
        }

    preds = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return {
        "loss": float(total_loss / total_samples),
        "accuracy": float(total_correct / total_samples),
        "predictions": preds,
        "targets": targets,
    }



def _save_history(output_dir: Path, history: dict[str, list[float]]) -> None:
    save_json(output_dir / "history.json", to_serializable(history))

    csv_path = output_dir / "history.csv"
    keys = list(history.keys())
    rows = zip(*[history[key] for key in keys])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)

    plot_loss_curves(history, output_dir / "loss_curve.png")
    plot_accuracy_curves(history, output_dir / "accuracy_curve.png")
    plot_lr_curve(history, output_dir / "lr_curve.png")



def train_model(
    model,
    repository: ImageRepository,
    train_records: Sequence[ImageRecord],
    val_records: Sequence[ImageRecord],
    mean: np.ndarray,
    std: np.ndarray,
    input_shape: Sequence[int],
    class_names: Sequence[str],
    output_dir: str | Path,
    config: TrainingConfig,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    best_dir = ensure_dir(output_dir / "best")
    last_dir = ensure_dir(output_dir / "last")

    model_parameters = list(model.parameters())
    optimizer = SGD(model_parameters, lr=config.learning_rate, momentum=config.momentum)
    scheduler = ExponentialLRScheduler(optimizer, gamma=config.lr_decay)

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_objective": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    train_start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_start = time.time()
        current_lr = optimizer.get_lr()
        train_loss_sum = 0.0
        train_obj_sum = 0.0
        train_correct = 0
        train_samples = 0

        aug_rng = np.random.default_rng(config.seed * 1000 + epoch)
        for batch_records in batch_iterator(
            train_records,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed + epoch,
        ):
            x_batch, y_batch = repository.records_to_batch(
                batch_records,
                mean=mean,
                std=std,
                flatten=True,
                augment=config.use_train_augmentation,
                rng=aug_rng,
                hflip_prob=config.train_hflip_prob,
                vflip_prob=config.train_vflip_prob,
                rotate_prob=config.train_rotate_prob,
            )
            x_tensor = Tensor(x_batch, requires_grad=False)

            optimizer.zero_grad()
            logits = model(x_tensor)
            data_loss = cross_entropy(logits, y_batch)
            reg_loss = l2_regularization(
                model.parameters(),
                coefficient=config.weight_decay,
                include_bias=config.include_bias_in_weight_decay,
            )
            loss = data_loss + reg_loss
            loss.backward()
            clip_grad_norm(model_parameters, config.max_grad_norm)
            optimizer.step()

            predictions = logits.data.argmax(axis=1)
            batch_size = x_batch.shape[0]
            train_loss_sum += data_loss.item() * batch_size
            train_obj_sum += loss.item() * batch_size
            train_correct += int((predictions == y_batch).sum())
            train_samples += int(batch_size)

        train_loss = float(train_loss_sum / max(train_samples, 1))
        train_objective = float(train_obj_sum / max(train_samples, 1))
        train_acc = float(train_correct / max(train_samples, 1))

        val_metrics = evaluate_split(
            model=model,
            repository=repository,
            records=val_records,
            mean=mean,
            std=std,
            batch_size=config.batch_size,
        )
        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["accuracy"])

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_objective"].append(train_objective)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rate"].append(current_lr)

        checkpoint_meta = {
            "class_names": list(class_names),
            "input_shape": list(input_shape),
            "normalization": {"mean": mean.tolist(), "std": std.tolist()},
            "epoch": epoch,
            "training_config": asdict(config),
            "history": history,
        }
        if extra_metadata:
            checkpoint_meta.update(extra_metadata)

        improved = val_acc > (best_val_acc + config.early_stopping_min_delta)
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_meta["best_val_accuracy"] = best_val_acc
            checkpoint_meta["best_epoch"] = best_epoch
            save_checkpoint(best_dir, model, checkpoint_meta)
        else:
            epochs_without_improvement += 1

        checkpoint_meta["best_val_accuracy"] = best_val_acc
        checkpoint_meta["best_epoch"] = best_epoch
        save_checkpoint(last_dir, model, checkpoint_meta)
        _save_history(output_dir, history)

        epoch_time = time.time() - epoch_start
        print(
            f"[Epoch {epoch:03d}/{config.epochs:03d}] "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={format_seconds(epoch_time)}"
        )

        if config.early_stopping_patience > 0 and epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"[Info] Early stopping triggered at epoch {epoch}. "
                f"No val_acc improvement for {epochs_without_improvement} epochs."
            )
            break

        scheduler.step()

    total_time = time.time() - train_start
    summary = {
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "actual_epochs": int(len(history["epoch"])),
        "total_training_time_seconds": float(total_time),
        "history": history,
        "best_checkpoint_dir": str(best_dir),
        "last_checkpoint_dir": str(last_dir),
    }
    if extra_metadata:
        summary.update(extra_metadata)
    save_json(output_dir / "training_summary.json", to_serializable(summary))
    return summary
