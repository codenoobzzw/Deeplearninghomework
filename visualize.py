from __future__ import annotations

import argparse
from pathlib import Path

from src.checkpoint import load_checkpoint
from src.utils import ensure_dir
from src.visualization import save_weight_grid



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize first-layer weights of a trained EuroSAT MLP.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="best checkpoint 目录")
    parser.add_argument("--output-dir", type=str, required=True, help="可视化输出目录")
    parser.add_argument("--max-filters", type=int, default=64, help="最多展示多少个隐藏单元的第一层权重")
    parser.add_argument("--cols", type=int, default=8, help="每行展示多少张")
    return parser



def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)

    model, metadata = load_checkpoint(args.checkpoint_dir)
    input_shape = metadata.get("input_shape")
    if input_shape is None:
        raise ValueError("checkpoint meta 中缺少 input_shape，无法将第一层权重还原为图像。")

    save_weight_grid(
        first_layer_weight=model.first_layer_weight(),
        input_shape=input_shape,
        output_path=output_dir / "first_layer_weights.png",
        max_filters=args.max_filters,
        cols=args.cols,
    )
    print(f"Saved first-layer weight visualization to: {output_dir / 'first_layer_weights.png'}")


if __name__ == "__main__":
    main()
