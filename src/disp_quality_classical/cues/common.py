from __future__ import annotations

import cv2
import numpy as np


def valid_roi(disparity: np.ndarray, roi_mask: np.ndarray, invalid_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    roi = np.asarray(roi_mask) > 0
    valid = roi & np.isfinite(disparity) & (np.asarray(disparity) > invalid_value)
    return roi, valid


def gradient_magnitude(image: np.ndarray, kernel: int = 3) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=kernel)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=kernel)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def safe_percentile(values: np.ndarray, q: float, epsilon: float = 1e-6) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return epsilon
    return max(float(np.percentile(values, q)), epsilon)
