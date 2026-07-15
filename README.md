# Classical Disparity Reliability — Thesis-Aligned Repository

This repository implements the final classical no-reference method described in the thesis **Depth Data Quality and Reliability Assessment for Warehouse Environments**.

## Inputs

- rectified intensity image,
- disparity map in pixel-disparity units,
- binary ROI mask supplied by the upstream segmentation stage.

The same code can be applied to sensor disparity and NVIDIA FoundationStereo (NFS) disparity. The repository assumes that NFS disparity is already aligned and represented in the same pixel-disparity scale as the sensor output. ToF reference disparity and sensor confidence are not inputs to the cue fusion.

## Seven cues

1. validity,
2. local smoothness / surface variation,
3. multiscale consistency,
4. edge coherence,
5. depth sensitivity,
6. hole clustering,
7. local outlier.

The final configuration is stored in:

```text
configs/thesis_final.yaml
```

The cue weights are:

```text
validity          0.18
smoothness        0.22
multiscale        0.12
edge coherence    0.14
depth sensitivity 0.14
hole clustering   0.14
local outlier     0.06
```

The ROI score uses the reported coefficients `0.55/0.15/0.15/0.10/0.05` for mean dense reliability, valid ratio, hole structure, smoothness, and stability.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Run one scene

```bash
dq-classical \
  --config configs/thesis_final.yaml \
  --intensity /path/to/intensity_left.png \
  --sensor-disparity /path/to/disparity.png \
  --nfs-disparity /path/to/nfs_disp.npy \
  --roi-mask /path/to/roi_mask.png \
  --output-dir outputs/example
```

The command writes dense cue maps, fused reliability maps, ROI scores, and JSON summaries.

## Design decisions

- Disparity arrays are not silently resized because resizing can change disparity units.
- NFS and sensor disparity use the same scale.
- Sensor confidence remains an independent baseline and is not fused.
- The scalar hole cue is broadcast across the ROI and the fused map is validity-masked, matching the thesis equations.
- The repository begins after ROI generation; the company-internal YOLO model is not distributed here.
