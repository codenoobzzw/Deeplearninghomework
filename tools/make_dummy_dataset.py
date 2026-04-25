from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]



def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic dataset with EuroSAT-like folder names.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--images-per-class", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_colors = {
        "AnnualCrop": np.array([160, 140, 80], dtype=np.uint8),
        "Forest": np.array([35, 110, 45], dtype=np.uint8),
        "HerbaceousVegetation": np.array([110, 165, 80], dtype=np.uint8),
        "Highway": np.array([120, 120, 120], dtype=np.uint8),
        "Industrial": np.array([135, 135, 145], dtype=np.uint8),
        "Pasture": np.array([150, 180, 100], dtype=np.uint8),
        "PermanentCrop": np.array([125, 150, 70], dtype=np.uint8),
        "Residential": np.array([180, 175, 170], dtype=np.uint8),
        "River": np.array([40, 95, 170], dtype=np.uint8),
        "SeaLake": np.array([55, 125, 200], dtype=np.uint8),
    }

    for class_name in CLASS_NAMES:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        base = base_colors[class_name].astype(np.int16)
        for idx in range(args.images_per_class):
            image = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
            noise = rng.integers(-25, 26, size=image.shape, dtype=np.int16)
            image[...] = np.clip(base + noise, 0, 255).astype(np.uint8)

            if class_name in {"River", "SeaLake"}:
                rr = slice(args.image_size // 4, 3 * args.image_size // 4)
                cc = slice(args.image_size // 3, 2 * args.image_size // 3)
                image[rr, cc, 2] = np.clip(image[rr, cc, 2].astype(np.int16) + 30, 0, 255).astype(np.uint8)
            elif class_name == "Forest":
                image[:, ::4, 1] = np.clip(image[:, ::4, 1].astype(np.int16) + 40, 0, 255).astype(np.uint8)
            elif class_name == "Highway":
                image[args.image_size // 2 - 2 : args.image_size // 2 + 2, :, :] = 200
            elif class_name == "Residential":
                for x in range(8, args.image_size, 16):
                    image[x : x + 4, x : x + 4, :] = 220

            Image.fromarray(image).save(class_dir / f"{class_name}_{idx:03d}.png")

    print(f"Synthetic dataset created at: {output_dir}")


if __name__ == "__main__":
    main()
