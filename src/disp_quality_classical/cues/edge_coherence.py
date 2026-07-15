from __future__ import annotations

import cv2
import numpy as np

from .common import gradient_magnitude, valid_roi


def compute_edge_coherence_cue(intensity, disparity, roi_mask, *, invalid_value=0.0, edge_percentile=75.0, dilation_kernel=3, dilation_iterations=1):
    _roi, valid = valid_roi(disparity, roi_mask, invalid_value)
    disparity = np.asarray(disparity, dtype=np.float32)
    cue = np.zeros_like(disparity, dtype=np.float32)
    if not valid.any():
        return cue, {"edge_iou": 0.0}
    image_gradient = gradient_magnitude(intensity, 3)
    disparity_gradient = gradient_magnitude(disparity, 3)
    image_threshold = float(np.percentile(image_gradient[valid], edge_percentile))
    disparity_threshold = float(np.percentile(disparity_gradient[valid], edge_percentile))
    image_edge = valid & (image_gradient >= image_threshold)
    disparity_edge = valid & (disparity_gradient >= disparity_threshold)
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    image_dilated = cv2.dilate(image_edge.astype(np.uint8), kernel, iterations=dilation_iterations).astype(bool)
    disparity_dilated = cv2.dilate(disparity_edge.astype(np.uint8), kernel, iterations=dilation_iterations).astype(bool)
    supported = (image_edge & disparity_dilated) | (disparity_edge & image_dilated)
    edge_union = image_edge | disparity_edge
    cue[valid] = 1.0
    cue[edge_union] = supported[edge_union].astype(np.float32)
    iou = float(supported[edge_union].mean()) if edge_union.any() else 1.0
    return cue, {"edge_iou": iou, "image_edge_pixels": int(image_edge.sum()), "disparity_edge_pixels": int(disparity_edge.sum())}
