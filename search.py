from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.data import ImageRepository, prepare_split
from src.nn import MLPClassifier
from src.searcher import generate_trials, load_search_config, merge_trial_config, sanitize_trial_name
from src.trainer import TrainingConfig, train_model
from src.utils import ensure_dir, parse_hidden_dims, save_json, set_seed, to_serializable



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hyper-parameter search for EuroSAT MLP.")
    parser.add_argument("--data-root", type=str, required=True, help="EuroSAT_RGB 根目录")
    parser.add_argument("--output-dir", type=str, required=True, help="搜索结果输出目录")
    parser.add_argument("--search-config", type=str, required=True, help="grid/random 搜索配置 JSON")
    parser.add_argument("--split-file", type=str, default=None, help="已有 split.json；若不存在则自动创建到 output-dir/split.json")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 search_config/base_config 中的 epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 search_config/base_config 中的 batch_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=None, help="仅运行前 N 个 trial，便于调试")
    return parser



def _save_results_csv(path: Path, results: list[dict]) -> None:
    if not results:
        return
    keys = sorted({key for result in results for key in result.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)



def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    split_file = Path(args.split_file) if args.split_file is not None else output_dir / "split.json"

    split, class_names, split_path = prepare_split(
        data_root=args.data_root,
        split_file=split_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    repository = ImageRepository(args.data_root, cache_images=args.cache_images, image_size=args.image_size)
    all_records = split["train"] + split["val"] + split["test"]
    if args.cache_images:
        print("[Info] Caching images into memory ...")
        repository.preload(all_records)

    input_shape = repository.infer_input_shape(split["train"])
    input_dim = int(np.prod(input_shape))
    mean, std = repository.compute_channel_stats(split["train"])

    search_config = load_search_config(args.search_config)
    base_config = search_config.get("base_config", {})
    trials = generate_trials(search_config, seed=args.seed)
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    save_json(
        output_dir / "dataset_info.json",
        {
            "data_root": str(Path(args.data_root).resolve()),
            "split_file": str(split_file.resolve()),
            "split_sizes": {k: len(v) for k, v in split.items()},
            "class_names": class_names,
            "input_shape": list(input_shape),
            "normalization": {"mean": mean.tolist(), "std": std.tolist()},
            "search_config": search_config,
            "num_trials": len(trials),
        },
    )

    override_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }

    results: list[dict] = []
    best_result: dict | None = None

    for trial_idx, trial in enumerate(trials, start=1):
        merged = merge_trial_config(base_config, trial, overrides=override_config)
        hidden_dims = parse_hidden_dims(merged.get("hidden_dims", 512))
        activation = str(merged.get("activation", "relu"))
        dropout = float(merged.get("dropout", 0.2))
        learning_rate = float(merged.get("learning_rate", 0.01))
        momentum = float(merged.get("momentum", 0.9))
        lr_decay = float(merged.get("lr_decay", 1.0))
        weight_decay = float(merged.get("weight_decay", 0.0005))
        max_grad_norm = float(merged.get("max_grad_norm", 5.0))
        use_train_augmentation = bool(merged.get("use_train_augmentation", True))
        train_hflip_prob = float(merged.get("train_hflip_prob", 0.5))
        train_vflip_prob = float(merged.get("train_vflip_prob", 0.5))
        train_rotate_prob = float(merged.get("train_rotate_prob", 0.5))
        early_stopping_patience = int(merged.get("early_stopping_patience", 6))
        early_stopping_min_delta = float(merged.get("early_stopping_min_delta", 1e-4))
        epochs = int(merged.get("epochs", 30))
        batch_size = int(merged.get("batch_size", 128))
        include_bias_in_weight_decay = bool(merged.get("include_bias_in_weight_decay", False))

        trial_name = sanitize_trial_name(
            trial_idx,
            {
                "learning_rate": learning_rate,
                "hidden_dims": hidden_dims,
                "activation": activation,
                "weight_decay": weight_decay,
            },
        )
        trial_output_dir = output_dir / trial_name
        print("\n" + "=" * 80)
        print(f"Starting trial {trial_idx}/{len(trials)}: {trial_name}")
        print(f"Config: {merged}")

        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=len(class_names),
            activation=activation,
            dropout=dropout,
        )
        train_cfg = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            seed=args.seed + trial_idx,
            use_train_augmentation=use_train_augmentation,
            train_hflip_prob=train_hflip_prob,
            train_vflip_prob=train_vflip_prob,
            train_rotate_prob=train_rotate_prob,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            include_bias_in_weight_decay=include_bias_in_weight_decay,
        )

        summary = train_model(
            model=model,
            repository=repository,
            train_records=split["train"],
            val_records=split["val"],
            mean=mean,
            std=std,
            input_shape=input_shape,
            class_names=class_names,
            output_dir=trial_output_dir,
            config=train_cfg,
            extra_metadata={
                "data_root": str(Path(args.data_root).resolve()),
                "split_file": str(split_file.resolve()),
                "search_trial_index": trial_idx,
                "search_trial_config": to_serializable(merged),
            },
        )

        result = {
            "trial_index": trial_idx,
            "trial_name": trial_name,
            "best_val_accuracy": float(summary["best_val_accuracy"]),
            "best_epoch": int(summary["best_epoch"]),
            "best_checkpoint_dir": str(Path(summary["best_checkpoint_dir"]).resolve()),
            "hidden_dims": str(hidden_dims),
            "activation": activation,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "lr_decay": lr_decay,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "use_train_augmentation": use_train_augmentation,
            "train_hflip_prob": train_hflip_prob,
            "train_vflip_prob": train_vflip_prob,
            "train_rotate_prob": train_rotate_prob,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        results.append(result)
        results_sorted = sorted(results, key=lambda x: x["best_val_accuracy"], reverse=True)
        save_json(output_dir / "search_results.json", results_sorted)
        _save_results_csv(output_dir / "search_results.csv", results_sorted)

        if best_result is None or result["best_val_accuracy"] > best_result["best_val_accuracy"]:
            best_result = result
            save_json(output_dir / "best_trial.json", best_result)

    print("\nSearch finished.")
    if best_result is not None:
        print(f"Best trial: {best_result['trial_name']}")
        print(f"Best val accuracy: {best_result['best_val_accuracy']:.4f}")
        print(f"Best checkpoint: {best_result['best_checkpoint_dir']}")


if __name__ == "__main__":
    main()
