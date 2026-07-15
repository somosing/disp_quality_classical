from __future__ import annotations

import numpy as np

from .common import valid_roi


def compute_validity_cue(disparity: np.ndarray, roi_mask: np.ndarray, invalid_value: float = 0.0):
    roi, valid = valid_roi(disparity, roi_mask, invalid_value)
    cue = valid.astype(np.float32)
    ratio = float(valid.sum() / max(int(roi.sum()), 1))
    return cue, {"valid_ratio": ratio, "valid_pixels": int(valid.sum()), "roi_pixels": int(roi.sum())}
