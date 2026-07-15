from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .io import read_disparity, read_gray, read_roi
from .pipeline import evaluate_disparity_roi


def save_result(result: dict, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{prefix}_reliability.npy", result["reliability_map"])
    for name, cue in result["cue_maps"].items():
        np.save(output_dir / f"{prefix}_cue_{name}.npy", cue)
    payload = {"roi_score": result["roi_score"], "score_details": result["score_details"], "cue_details": result["cue_details"]}
    (output_dir / f"{prefix}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the thesis-aligned classical reliability method.")
    parser.add_argument("--config", type=Path, default=Path("configs/thesis_final.yaml"))
    parser.add_argument("--intensity", type=Path, required=True)
    parser.add_argument("--sensor-disparity", type=Path, required=True)
    parser.add_argument("--nfs-disparity", type=Path, default=None, help="Optional NFS disparity in the same pixel scale as sensor disparity.")
    parser.add_argument("--roi-mask", type=Path, required=True, help="ROI supplied by the upstream segmentation stage.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    intensity = read_gray(args.intensity)
    roi = read_roi(args.roi_mask)
    sensor = read_disparity(args.sensor_disparity)
    sensor_result = evaluate_disparity_roi(intensity, sensor, roi, cfg)
    save_result(sensor_result, args.output_dir, "sensor")

    print(json.dumps({"sensor_roi_score": sensor_result["roi_score"]}, indent=2))
    if args.nfs_disparity is not None:
        nfs = read_disparity(args.nfs_disparity)
        nfs_result = evaluate_disparity_roi(intensity, nfs, roi, cfg)
        save_result(nfs_result, args.output_dir, "nfs")
        print(json.dumps({
            "nfs_roi_score": nfs_result["roi_score"],
            "nfs_minus_sensor": nfs_result["roi_score"] - sensor_result["roi_score"],
        }, indent=2))


if __name__ == "__main__":
    main()
