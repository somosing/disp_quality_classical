from __future__ import annotations

import cv2
import numpy as np

from .common import gradient_magnitude, safe_percentile, valid_roi


def compute_smoothness_cue(intensity, disparity, roi_mask, *, invalid_value=0.0, edge_percentile=75.0, normalization_percentile=95.0, epsilon=1e-6):
    roi, valid = valid_roi(disparity, roi_mask, invalid_value)
    disparity = np.asarray(disparity, dtype=np.float32)
    cue = np.zeros_like(disparity, dtype=np.float32)
    if not valid.any():
        return cue, {"laplacian_mean": 0.0, "evaluated_pixels": 0}
    laplacian = np.abs(cv2.Laplacian(disparity, cv2.CV_32F, ksize=3))
    image_gradient = gradient_magnitude(intensity, 3)
    edge_threshold = float(np.percentile(image_gradient[valid], edge_percentile))
    evaluated = valid & (image_gradient < edge_threshold)
    if not evaluated.any():
        evaluated = valid
    scale = safe_percentile(laplacian[evaluated], normalization_percentile, epsilon)
    cue[evaluated] = 1.0 - np.clip(laplacian[evaluated] / scale, 0.0, 1.0)
    return cue, {"laplacian_mean": float(laplacian[evaluated].mean()), "evaluated_pixels": int(evaluated.sum())}
