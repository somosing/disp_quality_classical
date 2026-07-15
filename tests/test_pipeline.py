from pathlib import Path

import numpy as np

from disp_quality_classical.config import load_config
from disp_quality_classical.pipeline import evaluate_disparity_roi


def config():
    return load_config(Path(__file__).parents[1] / "configs" / "thesis_final.yaml")


def test_pipeline_shapes_and_invalid_masking():
    h, w = 64, 96
    intensity = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    disparity = np.full((h, w), 40.0, dtype=np.float32)
    disparity[20:30, 30:50] = 0.0
    roi = np.ones((h, w), dtype=np.uint8)
    result = evaluate_disparity_roi(intensity, disparity, roi, config())
    assert result["reliability_map"].shape == disparity.shape
    assert set(result["cue_maps"]) == {
        "validity", "smoothness", "multiscale", "edge_coherence",
        "depth_sensitivity", "hole_clustering", "local_outlier",
    }
    assert np.all(result["reliability_map"][disparity == 0] == 0)
    assert 0.0 <= result["roi_score"] <= 1.0


def test_same_pipeline_can_compare_two_sources():
    h, w = 48, 64
    intensity = np.zeros((h, w), dtype=np.uint8)
    roi = np.ones((h, w), dtype=np.uint8)
    sensor = np.full((h, w), 20.0, dtype=np.float32)
    sensor[:, ::5] = 0.0
    nfs = np.full((h, w), 20.0, dtype=np.float32)
    sensor_result = evaluate_disparity_roi(intensity, sensor, roi, config())
    nfs_result = evaluate_disparity_roi(intensity, nfs, roi, config())
    assert nfs_result["score_details"]["valid_ratio"] > sensor_result["score_details"]["valid_ratio"]
