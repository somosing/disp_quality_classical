from __future__ import annotations

import cv2
import numpy as np

from .common import valid_roi


def compute_hole_clustering_cue(disparity, roi_mask, *, invalid_value=0.0, alpha_fraction=1.0, alpha_largest=1.0, alpha_count=0.3, count_reference_area=500.0):
    roi, valid = valid_roi(disparity, roi_mask, invalid_value)
    cue = np.zeros_like(np.asarray(disparity, dtype=np.float32))
    if not roi.any():
        return cue, {"hole_fraction": 1.0, "largest_hole_fraction": 1.0, "hole_count_norm": 1.0, "global_score": 0.0}
    invalid = roi & ~valid
    roi_area = float(roi.sum())
    hole_fraction = float(invalid.sum() / roi_area)
    labels, _map, stats, _centroids = cv2.connectedComponentsWithStats(invalid.astype(np.uint8), connectivity=8)
    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, labels)]
    largest_fraction = float(max(areas, default=0) / roi_area)
    count_norm = min(float(len(areas) / max(roi_area / count_reference_area, 1.0)), 1.0)
    penalty = alpha_fraction * hole_fraction + alpha_largest * largest_fraction + alpha_count * count_norm
    score = float(1.0 / (1.0 + 3.0 * penalty))
    cue[roi] = score
    return cue, {"hole_fraction": hole_fraction, "largest_hole_fraction": largest_fraction, "hole_count": len(areas), "hole_count_norm": count_norm, "global_score": score}
