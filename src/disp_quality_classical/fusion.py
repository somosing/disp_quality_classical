from __future__ import annotations

import numpy as np


def fuse_cues(cue_maps: dict[str, np.ndarray], weights: dict[str, float], valid_roi_mask: np.ndarray) -> np.ndarray:
    """Weighted cue fusion followed by the valid-object mask from the thesis."""
    if not cue_maps:
        raise ValueError("At least one cue map is required.")
    shape = next(iter(cue_maps.values())).shape
    numerator = np.zeros(shape, dtype=np.float32)
    denominator = 0.0
    for name, cue in cue_maps.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        if cue.shape != shape:
            raise ValueError(f"Cue '{name}' has shape {cue.shape}; expected {shape}.")
        numerator += weight * cue.astype(np.float32)
        denominator += weight
    if denominator <= 0:
        raise ValueError("The sum of active cue weights must be positive.")
    reliability = numerator / denominator
    reliability *= np.asarray(valid_roi_mask, dtype=np.float32)
    return np.clip(reliability, 0.0, 1.0).astype(np.float32)
