from __future__ import annotations

import cv2
import numpy as np

from .common import valid_roi


def compute_local_outlier_cue(disparity, roi_mask, *, invalid_value=0.0, window=5, alpha=2.5):
    _roi, valid = valid_roi(disparity, roi_mask, invalid_value)
    disparity = np.asarray(disparity, dtype=np.float32)
    cue = np.zeros_like(disparity, dtype=np.float32)
    if not valid.any():
        return cue, {"absolute_deviation_mean": 0.0}
    filled = np.where(valid, disparity, 0.0).astype(np.float32)
    median = cv2.medianBlur(filled, window)
    deviation = np.abs(disparity - median)
    cue[valid] = 1.0 / (1.0 + alpha * deviation[valid])
    return cue, {"absolute_deviation_mean": float(deviation[valid].mean())}
