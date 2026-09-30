from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .anomaly import SyntheticAnomalyGenerator

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MVTecDataset(Dataset):
    def __init__(
        self,
        root: str,
        category: str,
        train: bool,
        image_size: int = 256,
        texture_root: str | None = None,
        anomaly_probability: float = 0.5,
        require_foreground: bool = True,
        require_texture: bool = True,
        foreground_root: str | None = None,
    ) -> None:
        self.root = Path(root) / category
        self.category = category
        self.train = train
        self.resize = (image_size, image_size)
        self.anomaly_probability = anomaly_probability
        self.require_foreground = require_foreground
        self.foreground_root = foreground_root
        pattern = "train/good/*.png" if train else "test/**/*.png"
        self.image_paths = sorted(self.root.glob(pattern))
        self.transform = _image_transform()
        self.anomaly = (
            SyntheticAnomalyGenerator(
                self.resize, texture_root, require_texture=require_texture
            )
            if train
            else None
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = _read_rgb(path, self.resize)
        if self.train:
            foreground = _mvtec_foreground(
                path,
                image,
                self.resize,
                self.category,
                require=self.require_foreground,
                foreground_root=self.foreground_root,
            )
            if np.random.rand() < self.anomaly_probability and self.anomaly is not None:
                anomaly, mask = self.anomaly.generate(image, foreground)
                target = int(np.any(mask))
            else:
                anomaly = image
                mask = np.zeros(self.resize, dtype=np.int64)
                target = 0
            return (
                self.transform(image),
                self.transform(anomaly),
                torch.from_numpy(mask),
                target,
            )

        mask, target = _mvtec_mask(path, self.resize)
        return self.transform(image), torch.from_numpy(mask), target


class VisADataset(Dataset):
    def __init__(
        self,
        root: str,
        category: str,
        train: bool,
        image_size: int = 256,
        texture_root: str | None = None,
        anomaly_probability: float = 0.5,
        require_foreground: bool = True,
        require_texture: bool = True,
        foreground_root: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.train = train
        self.resize = (image_size, image_size)
        self.anomaly_probability = anomaly_probability
        self.require_foreground = require_foreground
        self.foreground_root = foreground_root
        split_csv = self.root / "split_csv" / "1cls.csv"
        df = pd.read_csv(split_csv)
        split_name = "train" if train else "test"
        df = df[df["split"].str.contains(split_name)]
        df = df[df["object"] == category]
        self.image_paths = [self.root / p for p in df["image"].tolist()]
        self.labels = df["label"].tolist()
        self.transform = _image_transform()
        self.anomaly = (
            SyntheticAnomalyGenerator(
                self.resize, texture_root, require_texture=require_texture
            )
            if train
            else None
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = _read_rgb(path, self.resize)
        target = 0 if self.labels[idx] == "normal" else 1
        if self.train:
            foreground = _visa_foreground(
                path,
                image,
                self.resize,
                self.category,
                require=self.require_foreground,
                foreground_root=self.foreground_root,
            )
            if np.random.rand() < self.anomaly_probability and self.anomaly is not None:
                anomaly, mask = self.anomaly.generate(image, foreground)
                target = int(np.any(mask))
            else:
                anomaly = image
                mask = np.zeros(self.resize, dtype=np.int64)
                target = 0
            return (
                self.transform(image),
                self.transform(anomaly),
                torch.from_numpy(mask),
                target,
            )

        mask = (
            _visa_mask(path, self.resize)
            if target
            else np.zeros(self.resize, dtype=np.int64)
        )
        return self.transform(image), torch.from_numpy(mask), target


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    train_set = _build_dataset(config, train=True)
    val_set = _build_dataset(config, train=False)
    data_cfg = config["data"]
    train_loader = DataLoader(
        train_set,
        batch_size=data_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=True,
    )
    return train_loader, _build_loader(val_set, data_cfg)


def build_eval_dataloader(config: dict) -> DataLoader:
    return _build_loader(_build_dataset(config, train=False), config["data"])


def _build_dataset(config: dict, train: bool) -> Dataset:
    data_cfg = config["data"]
    dataset_name = data_cfg["dataset"].lower()
    dataset_cls = {"mvtec": MVTecDataset, "visa": VisADataset}[dataset_name]
    return dataset_cls(
        root=data_cfg["root"],
        category=data_cfg["category"],
        train=train,
        image_size=data_cfg.get("image_size", 256),
        texture_root=data_cfg.get("texture_root"),
        anomaly_probability=data_cfg.get("anomaly_probability", 0.5),
        require_foreground=data_cfg.get("require_foreground", True),
        require_texture=data_cfg.get("require_texture", True),
        foreground_root=data_cfg.get("foreground_root"),
    )


def _build_loader(dataset: Dataset, data_cfg: dict) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=True,
    )


def _image_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def _read_rgb(path: Path, resize: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, dsize=(resize[1], resize[0]))


def _foreground_from_image(image: np.ndarray, category: str) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, background = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    background = (background > 0).astype(np.int64)
    if category in {"carpet", "cable", "wood"}:
        return np.ones(image.shape[:2], dtype=np.int64)
    if category in {"toothbrush", "hazelnut", "metal_nut", "pill"}:
        return background
    return 1 - background


def _mvtec_foreground(
    path: Path,
    image: np.ndarray,
    resize: tuple[int, int],
    category: str,
    require: bool,
    foreground_root: str | Path | None = None,
) -> np.ndarray:
    if category in {"carpet", "grid", "leather", "tile", "wood"}:
        return np.ones(resize, dtype=np.int64)
    foreground_path = (
        Path(foreground_root) / category / "train" / "foreground" / path.name
        if foreground_root is not None
        else path.parent.parent / "foreground" / path.name
    )
    if not foreground_path.exists():
        if require:
            raise FileNotFoundError(
                f"Foreground mask is required for {path}: expected {foreground_path}"
            )
        return _foreground_from_image(image, category)
    foreground = cv2.imread(str(foreground_path), cv2.IMREAD_GRAYSCALE)
    if foreground is None:
        if require:
            raise ValueError(f"Foreground mask is unreadable: {foreground_path}")
        return _foreground_from_image(image, category)
    foreground = cv2.resize(foreground, dsize=(resize[1], resize[0]))
    foreground = (foreground > 0).astype(np.int64)
    if require and not np.any(foreground):
        raise ValueError(f"Foreground mask is empty: {foreground_path}")
    return foreground


def _mvtec_mask(path: Path, resize: tuple[int, int]) -> tuple[np.ndarray, int]:
    mask_path = Path(
        str(path).replace("test", "ground_truth").replace(".png", "_mask.png")
    )
    if not mask_path.exists():
        return np.zeros(resize, dtype=np.int64), 0
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, dsize=(resize[1], resize[0]))
    return (mask > 0).astype(np.int64), 1


def _visa_foreground(
    path: Path,
    image: np.ndarray,
    resize: tuple[int, int],
    category: str,
    require: bool,
    foreground_root: str | Path | None = None,
) -> np.ndarray:
    foreground_path = (
        Path(foreground_root)
        / category
        / "Data"
        / "Foreground"
        / path.parent.name
        / path.with_suffix(".png").name
        if foreground_root is not None
        else path.parent.parent.parent / "Foreground" / path.parent.name / path.name
    )
    if not foreground_path.exists():
        if require:
            raise FileNotFoundError(
                f"Foreground mask is required for {path}: expected {foreground_path}"
            )
        return _foreground_from_image(image, category)
    foreground = cv2.imread(str(foreground_path), cv2.IMREAD_GRAYSCALE)
    if foreground is None:
        if require:
            raise ValueError(f"Foreground mask is unreadable: {foreground_path}")
        foreground = _foreground_from_image(image, category)
    else:
        foreground = cv2.resize(foreground, dsize=(resize[1], resize[0]))
        foreground = (foreground > 0).astype(np.int64)
    if require and not np.any(foreground):
        raise ValueError(f"Foreground mask is empty: {foreground_path}")
    return foreground


def _visa_mask(path: Path, resize: tuple[int, int]) -> np.ndarray:
    mask_path = Path(str(path).replace("Images", "Masks").replace("JPG", "png"))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(resize, dtype=np.int64)
    mask = cv2.resize(mask, dsize=(resize[1], resize[0]))
    return (mask > 0).astype(np.int64)
