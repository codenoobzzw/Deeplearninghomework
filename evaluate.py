from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.checkpoint import load_checkpoint
from src.data import ImageRepository, load_split_file
from src.metrics import confusion_matrix, per_class_accuracy
from src.trainer import evaluate_split
from src.utils import ensure_dir, save_json, to_serializable
from src.visualization import plot_confusion_matrix, save_misclassified_grid



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained EuroSAT MLP checkpoint on the test split.")
    parser.add_argument("--data-root", type=str, required=True, help="EuroSAT_RGB 根目录")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="训练好的 best checkpoint 目录")
    parser.add_argument("--split-file", type=str, default=None, help="训练时保存的 split.json；若不填则尝试从 checkpoint meta 中读取")
    parser.add_argument("--output-dir", type=str, required=True, help="评估输出目录")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--image-size", type=int, default=None, help="可选：与训练保持一致的 resize 大小")
    parser.add_argument("--max-error-examples", type=int, default=16, help="可视化多少张测试错例")
    return parser



def _save_predictions_csv(path: Path, records, y_true, y_pred, class_names) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "true_label", "pred_label", "true_name", "pred_name", "correct"])
        for record, t, p in zip(records, y_true, y_pred):
            writer.writerow([
                record.relative_path,
                int(t),
                int(p),
                class_names[int(t)],
                class_names[int(p)],
                int(t == p),
            ])



def _format_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> str:
    width = max(max(len(name) for name in class_names), 8)
    header = " " * (width + 2) + " ".join(f"{name[:width]:>{width}}" for name in class_names)
    rows = [header]
    for idx, name in enumerate(class_names):
        values = " ".join(f"{int(v):>{width}d}" for v in cm[idx])
        rows.append(f"{name[:width]:>{width}} | {values}")
    return "\n".join(rows)



def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)

    model, metadata = load_checkpoint(args.checkpoint_dir)
    split_file = args.split_file or metadata.get("split_file")
    if not split_file:
        raise ValueError("没有提供 --split-file，且 checkpoint meta 中也没有保存 split_file。")

    split, class_names, _ = load_split_file(split_file)

    meta_input_shape = metadata.get("input_shape")
    default_image_size = None
    if args.image_size is not None:
        default_image_size = args.image_size
    elif meta_input_shape is not None and len(meta_input_shape) >= 2 and meta_input_shape[0] == meta_input_shape[1]:
        default_image_size = int(meta_input_shape[0])

    repository = ImageRepository(args.data_root, cache_images=args.cache_images, image_size=default_image_size)
    test_records = split["test"]
    if args.cache_images:
        print("[Info] Caching test images into memory ...")
        repository.preload(test_records)

    norm = metadata["normalization"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)

    metrics = evaluate_split(
        model=model,
        repository=repository,
        records=test_records,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
    )
    y_true = metrics["targets"]
    y_pred = metrics["predictions"]

    cm = confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    cls_acc = per_class_accuracy(cm)

    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
    num_errors_visualized = save_misclassified_grid(
        records=test_records,
        y_true=y_true,
        y_pred=y_pred,
        repository=repository,
        class_names=class_names,
        output_path=output_dir / "misclassified_examples.png",
        max_examples=args.max_error_examples,
    )
    _save_predictions_csv(output_dir / "predictions.csv", test_records, y_true, y_pred, class_names)

    results = {
        "test_loss": float(metrics["loss"]),
        "test_accuracy": float(metrics["accuracy"]),
        "num_test_samples": int(len(test_records)),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": {name: float(cls_acc[idx]) for idx, name in enumerate(class_names)},
        "num_errors_visualized": int(num_errors_visualized),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "split_file": str(Path(split_file).resolve()),
    }
    save_json(output_dir / "test_metrics.json", to_serializable(results))

    print(f"Test loss: {results['test_loss']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print("Per-class accuracy:")
    for name in class_names:
        print(f"  {name}: {results['per_class_accuracy'][name]:.4f}")
    print("\nConfusion matrix:")
    print(_format_confusion_matrix(cm, class_names))
    print(f"\nSaved evaluation results to: {output_dir}")


if __name__ == "__main__":
    main()
