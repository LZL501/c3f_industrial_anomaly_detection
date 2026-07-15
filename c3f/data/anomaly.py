from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def rand_perlin_2d_np(shape: tuple[int, int], res: tuple[int, int]) -> np.ndarray:
    def f(t: np.ndarray) -> np.ndarray:
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


class SyntheticAnomalyGenerator:
    def __init__(
        self,
        resize: tuple[int, int] = (256, 256),
        texture_root: str | None = None,
        structure_grid_size: int = 8,
        transparency_range: tuple[float, float] = (0.15, 1.0),
        perlin_scale: int = 6,
        min_perlin_scale: int = 0,
        perlin_noise_threshold: float = 0.5,
        require_texture: bool = False,
    ) -> None:
        self.resize = resize
        self.texture_paths = list(_iter_texture_paths(texture_root))
        if require_texture and not self.texture_paths:
            raise FileNotFoundError(
                f"No DTD texture images found under: {texture_root}"
            )
        self.structure_grid_size = structure_grid_size
        self.transparency_range = transparency_range
        self.perlin_scale = perlin_scale
        self.min_perlin_scale = min_perlin_scale
        self.perlin_noise_threshold = perlin_noise_threshold

    def generate(
        self, image: np.ndarray, foreground: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        foreground = (foreground > 0).astype(np.float32)
        if foreground.shape != self.resize:
            raise ValueError(
                f"Expected foreground shape {self.resize}, got {foreground.shape}"
            )
        if not np.any(foreground):
            raise ValueError(
                "Foreground mask is empty; refusing to create an off-object pseudo anomaly."
            )
        for _ in range(50):
            perlin_mask = self._perlin_mask()
            mask = (perlin_mask * foreground).astype(np.float32)
            if np.any(mask):
                break
        else:
            raise RuntimeError(
                "Could not sample a non-empty pseudo-anomaly mask inside the foreground."
            )
        mask3 = np.expand_dims(mask, axis=2)
        source = self._source(image)
        factor = np.random.uniform(*self.transparency_range)
        anomaly = factor * mask3 * source + (1 - factor) * mask3 * image
        anomaly = (1 - mask3) * image + anomaly
        return anomaly.astype(np.uint8), mask.astype(np.int64)

    def _perlin_mask(self) -> np.ndarray:
        scalex = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale)
        scaley = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale)
        noise = rand_perlin_2d_np(self.resize, (scalex, scaley))
        angle = float(np.random.uniform(-90, 90))
        center = (self.resize[1] / 2.0, self.resize[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        noise = cv2.warpAffine(
            noise, matrix, (self.resize[1], self.resize[0]), flags=cv2.INTER_LINEAR
        )
        return (noise > self.perlin_noise_threshold).astype(np.float32)

    def _source(self, image: np.ndarray) -> np.ndarray:
        if self.texture_paths and np.random.rand() < 0.5:
            return self._texture_source()
        return self._structure_source(image)

    def _texture_source(self) -> np.ndarray:
        for _ in range(min(10, len(self.texture_paths))):
            path = np.random.choice(self.texture_paths)
            texture = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if texture is not None:
                texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                return cv2.resize(
                    texture, dsize=(self.resize[1], self.resize[0])
                ).astype(np.float32)
        raise FileNotFoundError("No readable texture image found.")

    def _structure_source(self, image: np.ndarray) -> np.ndarray:
        source = self._random_augment(image)
        h, w = self.resize
        grid = self.structure_grid_size
        assert h % grid == 0 and w % grid == 0
        gh, gw = h // grid, w // grid
        patches = source.reshape(grid, gh, grid, gw, 3).transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(grid * grid, gh, gw, 3)
        np.random.shuffle(patches)
        return (
            patches.reshape(grid, grid, gh, gw, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(h, w, 3)
            .astype(np.float32)
        )

    def _random_augment(self, image: np.ndarray) -> np.ndarray:
        augmenters = (
            self._gamma_contrast,
            self._brightness,
            self._sharpness,
            self._hue_and_saturation,
            self._solarize,
            self._posterize,
            self._invert,
            self._autocontrast,
            self._equalize,
            self._rotate,
        )
        source = image.copy()
        for index in np.random.choice(len(augmenters), size=3, replace=False):
            source = augmenters[int(index)](source)
        return source

    @staticmethod
    def _gamma_contrast(image: np.ndarray) -> np.ndarray:
        gamma = np.random.uniform(0.5, 2.0, size=(1, 1, 3))
        adjusted = 255.0 * np.power(np.clip(image / 255.0, 0, 1), gamma)
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def _brightness(image: np.ndarray) -> np.ndarray:
        multiplier = np.random.uniform(0.8, 1.2, size=(1, 1, 3))
        offset = np.random.uniform(-30, 30, size=(1, 1, 3))
        return np.clip(image * multiplier + offset, 0, 255).astype(np.uint8)

    @staticmethod
    def _sharpness(image: np.ndarray) -> np.ndarray:
        factor = float(np.random.uniform(0.0, 2.0))
        return np.asarray(
            ImageEnhance.Sharpness(Image.fromarray(image)).enhance(factor)
        )

    @staticmethod
    def _hue_and_saturation(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + np.random.randint(-25, 26)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + np.random.randint(-50, 51), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    @staticmethod
    def _solarize(image: np.ndarray) -> np.ndarray:
        threshold = int(np.random.randint(32, 129))
        return np.asarray(
            ImageOps.solarize(Image.fromarray(image), threshold=threshold)
        )

    @staticmethod
    def _posterize(image: np.ndarray) -> np.ndarray:
        bits = int(np.random.randint(1, 9))
        return np.asarray(ImageOps.posterize(Image.fromarray(image), bits=bits))

    @staticmethod
    def _invert(image: np.ndarray) -> np.ndarray:
        return np.asarray(ImageOps.invert(Image.fromarray(image)))

    @staticmethod
    def _autocontrast(image: np.ndarray) -> np.ndarray:
        return np.asarray(ImageOps.autocontrast(Image.fromarray(image)))

    @staticmethod
    def _equalize(image: np.ndarray) -> np.ndarray:
        return np.asarray(ImageOps.equalize(Image.fromarray(image)))

    def _rotate(self, image: np.ndarray) -> np.ndarray:
        angle = float(np.random.uniform(-45, 45))
        center = (self.resize[1] / 2.0, self.resize[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (self.resize[1], self.resize[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )


def _iter_texture_paths(texture_root: str | None) -> Iterable[Path]:
    if not texture_root:
        return []
    root = Path(texture_root)
    if not root.exists():
        return []
    return [
        p
        for p in root.glob("*/*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
