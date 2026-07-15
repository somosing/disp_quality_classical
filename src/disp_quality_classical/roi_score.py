from __future__ import annotations

import numpy as np


def _mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    return float(selected.mean()) if selected.size else 0.0


def compute_roi_score(reliability_map, disparity, roi_mask, cue_maps, valid_ratio: float, cfg: dict):
    roi = np.asarray(roi_mask) > 0
    disparity = np.asarray(disparity, dtype=np.float32)
    invalid_value = float(cfg.get("invalid_disparity_value", 0.0))
    valid = roi & np.isfinite(disparity) & (disparity > invalid_value)
    if not roi.any():
        return 0.0, {"status": "empty_roi"}

    mean_reliability = _mean(reliability_map, roi)
    hole_mean = _mean(cue_maps["hole_clustering"], roi)
    smoothness_mean = _mean(cue_maps["smoothness"], valid)
    disparity_std = float(np.std(disparity[valid])) if valid.any() else 0.0
    instability_scale = float(cfg["parameters"].get("roi_instability_scale_px", 30.0))
    instability = min(disparity_std / max(instability_scale, 1e-6), 1.0)

    weights = cfg["roi_score"]
    score = (
        float(weights["mean_reliability"]) * mean_reliability
        + float(weights["valid_ratio"]) * valid_ratio
        + float(weights["hole_clustering"]) * hole_mean
        + float(weights["smoothness"]) * smoothness_mean
        + float(weights["stability"]) * (1.0 - instability)
    )
    score = float(np.clip(score, 0.0, 1.0))
    return score, {
        "status": "ok",
        "mean_reliability": mean_reliability,
        "valid_ratio": float(valid_ratio),
        "hole_clustering_mean": hole_mean,
        "smoothness_mean": smoothness_mean,
        "disparity_std_px": disparity_std,
        "instability_penalty": instability,
        "final_score": score,
    }
