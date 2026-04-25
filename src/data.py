from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from .utils import ensure_dir, load_json, save_json


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    relative_path: str
    label: int
    class_name: str



def discover_dataset(data_root: str | Path) -> tuple[list[ImageRecord], list[str]]:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_root}")

    class_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"在 {data_root} 下没有发现类别文件夹。")

    class_names = sorted([p.name for p in class_dirs])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    records: list[ImageRecord] = []
    for class_name in class_names:
        folder = data_root / class_name
        image_paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        for image_path in sorted(image_paths):
            records.append(
                ImageRecord(
                    relative_path=str(image_path.relative_to(data_root)).replace("\\", "/"),
                    label=class_to_idx[class_name],
                    class_name=class_name,
                )
            )

    if not records:
        raise RuntimeError(f"在 {data_root} 下没有发现支持的图像文件。")

    return records, class_names



def stratified_split(
    records: Sequence[ImageRecord],
    num_classes: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[ImageRecord]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1。")

    rng = np.random.default_rng(seed)
    per_class: dict[int, list[ImageRecord]] = {i: [] for i in range(num_classes)}
    for record in records:
        per_class[record.label].append(record)

    split = {"train": [], "val": [], "test": []}
    for class_idx in range(num_classes):
        class_records = list(per_class[class_idx])
        if not class_records:
            continue
        indices = np.arange(len(class_records))
        rng.shuffle(indices)
        shuffled = [class_records[i] for i in indices]

        n = len(shuffled)
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val

        # 尽量保证每个划分非空（当样本量足够时）
        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
            n_test = n - n_train - n_val
            if n_test <= 0:
                n_test = 1
                if n_train >= n_val and n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1

        split["train"].extend(shuffled[:n_train])
        split["val"].extend(shuffled[n_train : n_train + n_val])
        split["test"].extend(shuffled[n_train + n_val :])

    return split



def save_split_file(
    split_path: str | Path,
    split: dict[str, Sequence[ImageRecord]],
    class_names: Sequence[str],
    seed: int,
    ratios: dict[str, float],
) -> None:
    payload = {
        "class_names": list(class_names),
        "seed": int(seed),
        "ratios": ratios,
        "splits": {name: [asdict(record) for record in records] for name, records in split.items()},
    }
    save_json(split_path, payload)



def load_split_file(split_path: str | Path) -> tuple[dict[str, list[ImageRecord]], list[str], dict]:
    payload = load_json(split_path)
    class_names = payload["class_names"]
    split = {
        name: [ImageRecord(**record) for record in payload["splits"][name]]
        for name in ["train", "val", "test"]
    }
    meta = {k: v for k, v in payload.items() if k not in {"class_names", "splits"}}
    return split, class_names, meta



def prepare_split(
    data_root: str | Path,
    split_file: str | Path | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, list[ImageRecord]], list[str], Path | None]:
    if split_file is not None and Path(split_file).exists():
        split, class_names, _ = load_split_file(split_file)
        return split, class_names, Path(split_file)

    records, class_names = discover_dataset(data_root)
    split = stratified_split(
        records=records,
        num_classes=len(class_names),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_path_obj = Path(split_file) if split_file is not None else None
    if split_path_obj is not None:
        ensure_dir(split_path_obj.parent)
        save_split_file(
            split_path=split_path_obj,
            split=split,
            class_names=class_names,
            seed=seed,
            ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
        )
    return split, class_names, split_path_obj


class ImageRepository:
    def __init__(self, data_root: str | Path, cache_images: bool = False, image_size: int | None = None) -> None:
        self.data_root = Path(data_root)
        self.cache_images = bool(cache_images)
        self.image_size = int(image_size) if image_size is not None else None
        self._cache: dict[str, np.ndarray] = {}

    def absolute_path(self, record: ImageRecord) -> Path:
        return self.data_root / record.relative_path

    def load_image(self, record: ImageRecord) -> np.ndarray:
        cache_key = record.relative_path
        if self.cache_images and cache_key in self._cache:
            return self._cache[cache_key]

        image_path = self.absolute_path(record)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if self.image_size is not None:
                img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
            array = np.asarray(img, dtype=np.uint8)

        if self.cache_images:
            self._cache[cache_key] = array
        return array

    def preload(self, records: Iterable[ImageRecord]) -> None:
        seen: set[str] = set()
        for record in records:
            if record.relative_path in seen:
                continue
            self.load_image(record)
            seen.add(record.relative_path)

    def infer_input_shape(self, records: Sequence[ImageRecord]) -> tuple[int, int, int]:
        if not records:
            raise ValueError("records 为空，无法推断输入形状。")
        image = self.load_image(records[0])
        return tuple(int(v) for v in image.shape)

    def compute_channel_stats(self, records: Sequence[ImageRecord]) -> tuple[np.ndarray, np.ndarray]:
        if not records:
            raise ValueError("records 为空，无法计算均值和方差。")

        channel_sum = np.zeros(3, dtype=np.float64)
        channel_sq_sum = np.zeros(3, dtype=np.float64)
        num_pixels = 0

        for record in records:
            image = self.load_image(record).astype(np.float32) / 255.0
            flat = image.reshape(-1, 3)
            channel_sum += flat.sum(axis=0)
            channel_sq_sum += np.square(flat).sum(axis=0)
            num_pixels += flat.shape[0]

        mean = channel_sum / float(num_pixels)
        var = channel_sq_sum / float(num_pixels) - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-12))
        return mean.astype(np.float32), std.astype(np.float32)

    def records_to_batch(
        self,
        records: Sequence[ImageRecord],
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        flatten: bool = True,
        augment: bool = False,
        rng: np.random.Generator | None = None,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate_prob: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not records:
            raise ValueError("一个 batch 中至少需要包含一张图像。")

        if augment and rng is None:
            rng = np.random.default_rng()

        images: list[np.ndarray] = []
        labels: list[int] = []
        for record in records:
            image = self.load_image(record).astype(np.float32) / 255.0
            if augment and rng is not None:
                image = augment_image(
                    image=image,
                    rng=rng,
                    hflip_prob=hflip_prob,
                    vflip_prob=vflip_prob,
                    rotate_prob=rotate_prob,
                )
            if mean is not None and std is not None:
                image = (image - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
            if flatten:
                image = image.reshape(-1)
            images.append(image.astype(np.float32))
            labels.append(int(record.label))

        return np.stack(images, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def augment_image(
    image: np.ndarray,
    rng: np.random.Generator,
    hflip_prob: float,
    vflip_prob: float,
    rotate_prob: float,
) -> np.ndarray:
    out = image
    if rng.random() < rotate_prob:
        k = int(rng.integers(1, 4))
        out = np.rot90(out, k=k, axes=(0, 1))
    if rng.random() < hflip_prob:
        out = out[:, ::-1, :]
    if rng.random() < vflip_prob:
        out = out[::-1, :, :]
    return np.ascontiguousarray(out, dtype=np.float32)



def batch_iterator(
    records: Sequence[ImageRecord],
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> Iterator[list[ImageRecord]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")

    indices = np.arange(len(records))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield [records[i] for i in batch_indices]
