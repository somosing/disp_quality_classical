from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_disparity(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise FileNotFoundError(path)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D disparity: {path}")
    return array.astype(np.float32)


def read_roi(path: str | Path) -> np.ndarray:
    return read_gray(path) > 0
