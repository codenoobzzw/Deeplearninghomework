from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data import ImageRepository, prepare_split
from src.nn import MLPClassifier
from src.trainer import TrainingConfig, train_model
from src.utils import ensure_dir, parse_hidden_dims, save_json, set_seed, to_serializable



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an MLP classifier on EuroSAT from scratch (NumPy + custom autograd).")
    parser.add_argument("--data-root", type=str, required=True, help="EuroSAT_RGB 根目录，内部应为按类别分文件夹的图像。")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录。")
    parser.add_argument("--split-file", type=str, default=None, help="已有划分文件；若不存在则自动创建。默认保存到 output-dir/split.json")
    parser.add_argument("--hidden-dims", type=str, default="512", help="隐藏层大小，如 512 或 512,256")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"], help="激活函数")
    parser.add_argument("--dropout", type=float, default=0.2, help="隐藏层 dropout 概率")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--lr-decay", type=float, default=0.98, help="每个 epoch 的学习率衰减系数")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="L2 正则化系数")
    parser.add_argument("--max-grad-norm", type=float, default=5.0, help="梯度裁剪阈值，<=0 表示禁用")
    parser.add_argument("--disable-train-augmentation", action="store_true", help="关闭训练集数据增强")
    parser.add_argument("--train-hflip-prob", type=float, default=0.5, help="训练集水平翻转概率")
    parser.add_argument("--train-vflip-prob", type=float, default=0.5, help="训练集垂直翻转概率")
    parser.add_argument("--train-rotate-prob", type=float, default=0.5, help="训练集随机90度旋转概率")
    parser.add_argument("--early-stopping-patience", type=int, default=6, help="验证集准确率无提升时的早停耐心轮数，<=0 表示禁用")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4, help="早停判定的最小提升阈值")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--cache-images", action="store_true", help="将图像缓存到内存中，加快训练（占用更多内存）")
    parser.add_argument("--image-size", type=int, default=None, help="可选：训练前将图像 resize 到固定大小")
    parser.add_argument("--include-bias-in-weight-decay", action="store_true", help="是否对 bias 也施加 L2 正则")
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=len(class_names),
        activation=args.activation,
        dropout=args.dropout,
    )

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        use_train_augmentation=not args.disable_train_augmentation,
        train_hflip_prob=args.train_hflip_prob,
        train_vflip_prob=args.train_vflip_prob,
        train_rotate_prob=args.train_rotate_prob,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        include_bias_in_weight_decay=args.include_bias_in_weight_decay,
    )

    experiment_config = {
        "data_root": str(Path(args.data_root).resolve()),
        "output_dir": str(output_dir.resolve()),
        "split_file": str(split_file.resolve()),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "class_names": class_names,
        "input_shape": list(input_shape),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "model": model.config(),
        "training_config": vars(args),
    }
    save_json(output_dir / "experiment_config.json", to_serializable(experiment_config))

    summary = train_model(
        model=model,
        repository=repository,
        train_records=split["train"],
        val_records=split["val"],
        mean=mean,
        std=std,
        input_shape=input_shape,
        class_names=class_names,
        output_dir=output_dir,
        config=config,
        extra_metadata={
            "data_root": str(Path(args.data_root).resolve()),
            "split_file": str(split_file.resolve()),
        },
    )

    print("\nTraining finished.")
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f} (epoch {summary['best_epoch']})")
    print(f"Best checkpoint: {summary['best_checkpoint_dir']}")


if __name__ == "__main__":
    main()
