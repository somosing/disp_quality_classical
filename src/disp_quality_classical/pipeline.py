from __future__ import annotations

import numpy as np

from .cues import (
    compute_depth_sensitivity_cue,
    compute_edge_coherence_cue,
    compute_hole_clustering_cue,
    compute_local_outlier_cue,
    compute_multiscale_cue,
    compute_smoothness_cue,
    compute_validity_cue,
)
from .fusion import fuse_cues
from .roi_score import compute_roi_score


def evaluate_disparity_roi(intensity: np.ndarray, disparity: np.ndarray, roi_mask: np.ndarray, cfg: dict) -> dict:
    """Evaluate one disparity source with the exact seven-cue thesis formulation."""
    if disparity.ndim != 2 or intensity.ndim != 2 or roi_mask.ndim != 2:
        raise ValueError("Intensity, disparity, and ROI must be aligned 2D arrays.")
    if intensity.shape != disparity.shape or roi_mask.shape != disparity.shape:
        raise ValueError("Intensity, disparity, and ROI must have identical shapes; disparity is not resized.")

    invalid = float(cfg.get("invalid_disparity_value", 0.0))
    p = cfg["parameters"]
    validity, validity_details = compute_validity_cue(disparity, roi_mask, invalid)
    smoothness, smoothness_details = compute_smoothness_cue(
        intensity, disparity, roi_mask, invalid_value=invalid,
        edge_percentile=float(p["edge_percentile"]),
        normalization_percentile=float(p["normalization_percentile"]),
        epsilon=float(p["epsilon"]),
    )
    multiscale, multiscale_details = compute_multiscale_cue(
        intensity, disparity, roi_mask, invalid_value=invalid,
        small_kernel=int(p["multiscale_small_kernel"]),
        large_kernel=int(p["multiscale_large_kernel"]),
        edge_percentile=float(p["edge_percentile"]),
        normalization_percentile=float(p["normalization_percentile"]),
        epsilon=float(p["epsilon"]),
    )
    edge, edge_details = compute_edge_coherence_cue(
        intensity, disparity, roi_mask, invalid_value=invalid,
        edge_percentile=float(p["edge_percentile"]),
        dilation_kernel=int(p["edge_dilation_kernel"]),
        dilation_iterations=int(p["edge_dilation_iterations"]),
    )
    depth, depth_details = compute_depth_sensitivity_cue(
        disparity, roi_mask, invalid_value=invalid,
        window=int(p["depth_window"]), alpha=float(p["depth_alpha"]), epsilon=float(p["epsilon"]),
    )
    hole, hole_details = compute_hole_clustering_cue(
        disparity, roi_mask, invalid_value=invalid,
        alpha_fraction=float(p["hole_alpha_fraction"]),
        alpha_largest=float(p["hole_alpha_largest"]),
        alpha_count=float(p["hole_alpha_count"]),
        count_reference_area=float(p["hole_count_reference_area"]),
    )
    outlier, outlier_details = compute_local_outlier_cue(
        disparity, roi_mask, invalid_value=invalid,
        window=int(p["local_outlier_window"]), alpha=float(p["local_outlier_alpha"]),
    )

    cue_maps = {
        "validity": validity,
        "smoothness": smoothness,
        "multiscale": multiscale,
        "edge_coherence": edge,
        "depth_sensitivity": depth,
        "hole_clustering": hole,
        "local_outlier": outlier,
    }
    valid_roi = validity > 0
    reliability = fuse_cues(cue_maps, cfg["weights"], valid_roi)
    roi_score, score_details = compute_roi_score(
        reliability, disparity, roi_mask, cue_maps, validity_details["valid_ratio"], cfg
    )
    return {
        "cue_maps": cue_maps,
        "cue_details": {
            "validity": validity_details,
            "smoothness": smoothness_details,
            "multiscale": multiscale_details,
            "edge_coherence": edge_details,
            "depth_sensitivity": depth_details,
            "hole_clustering": hole_details,
            "local_outlier": outlier_details,
        },
        "reliability_map": reliability,
        "roi_score": roi_score,
        "score_details": score_details,
    }
