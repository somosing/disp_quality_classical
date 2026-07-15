from __future__ import annotations

import cv2
import numpy as np

from .common import valid_roi


def compute_depth_sensitivity_cue(disparity, roi_mask, *, invalid_value=0.0, window=5, alpha=400.0, epsilon=1e-6):
    _roi, valid = valid_roi(disparity, roi_mask, invalid_value)
    disparity = np.asarray(disparity, dtype=np.float32)
    cue = np.zeros_like(disparity, dtype=np.float32)
    if not valid.any():
        return cue, {"risk_mean": 0.0}
    filled = np.where(valid, disparity, 0.0).astype(np.float32)
    mean = cv2.blur(filled, (window, window))
    mean_sq = cv2.blur(filled * filled, (window, window))
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    risk = np.zeros_like(disparity, dtype=np.float32)
    risk[valid] = local_std[valid] / (disparity[valid] ** 2 + epsilon)
    cue[valid] = 1.0 / (1.0 + alpha * risk[valid])
    return cue, {"risk_mean": float(risk[valid].mean()), "local_std_mean": float(local_std[valid].mean())}
