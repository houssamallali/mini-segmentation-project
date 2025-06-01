#!/usr/bin/env python
# fuse_gap.py ---------------------------------------------------------------
# Measure the opening distance d of a high-rupturing-capacity fuse in X-ray
# footage.  Re-implementation of the ENSM-SE image-processing mini-project.
#
# Requirements --------------------------------------------------------------
#   pip install opencv-python numpy matplotlib scipy
#
# Usage ---------------------------------------------------------------------
#   python fuse_gap.py --video Camera_15_04_58.mp4
#   (outputs distance.csv  +  distance_plot.png)
#
# ---------------------------------------------------------------------------

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_opening, label

# ---------------------------------------------------------------------------
# Hyper-parameters 
# ---------------------------------------------------------------------------
BLUR_KERNEL_SIZE = 5           # Gaussian blur radius (pixels) 
MORPH_STRUCT_SIZE = 3          # 3×3 structuring element for opening/closing
MIN_COMP_AREA = 500            # pixels – smallest blob kept as "fuse half"
H_MM_REAL = 2.0                # real height H in millimetres
# ---------------------------------------------------------------------------


def read_video_gray(path: str) -> list[np.ndarray]:
    """
    Load every frame of the video as an 8-bit grayscale ndarray.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
               if frame.ndim == 3 else frame
        frames.append(gray)
    cap.release()
    print(f"[INFO] {len(frames)} frames loaded")
    return frames


def segment_fuse(img: np.ndarray) -> np.ndarray:
    """
    Return a clean binary mask of the fuse (1 = fuse pixel, 0 = background).
    """
    blurred = cv2.GaussianBlur(img, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    _, bin_inv = cv2.threshold(
        blurred, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # Morphological cleanup
    struct = np.ones((MORPH_STRUCT_SIZE, MORPH_STRUCT_SIZE), np.uint8)
    mask = binary_closing(bin_inv, structure=struct).astype(bool)
    mask = binary_opening(mask, structure=struct).astype(bool)
    return mask.astype(np.uint8)


def estimate_H_pixels(mask: np.ndarray) -> float:
    """
    Measure the vertical thickness (H) of the intact fuse on the first frame.
    We take the median over the central 50 % columns to avoid edge artefacts.
    """
    cols = mask.shape[1]
    col_range = slice(cols // 4, 3 * cols // 4)
    heights = [
        np.count_nonzero(mask[:, c]) for c in range(*col_range.indices(cols))
        if np.count_nonzero(mask[:, c])
    ]
    if not heights:
        raise RuntimeError("Could not measure H – segmentation failed.")
    H_pixels = float(np.median(heights))
    print(f"[INFO] H ≈ {H_pixels:.1f} px (should correspond to 2 mm)")
    return H_pixels


def gap_distance(mask: np.ndarray) -> float:
    """
    Compute the horizontal gap between the two fuse halves on *this* frame.
    Returns 0 if the fuse is still intact (single component).
    """
    lbl, n = label(mask)
    if n < 2:
        return 0.0

    # Keep the two largest components (the fuse halves)
    areas = [(lbl == i).sum() for i in range(1, n + 1)]
    idx = np.argsort(areas)[-2:]
    comps = [np.column_stack(np.where(lbl == i + 1)) for i in idx]

    # Ensure left component comes first
    comps.sort(key=lambda pts: pts[:, 1].mean())

    right_edge_left  = comps[0][:, 1].max()
    left_edge_right  = comps[1][:, 1].min()
    gap_px = max(left_edge_right - right_edge_left - 1, 0)
    return float(gap_px)


def process_video(path: str) -> list[float]:
    frames = read_video_gray(path)

    # ---- calibration using the first frame --------------------------------
    first_mask = segment_fuse(frames[0])
    H_px = estimate_H_pixels(first_mask)
    px_per_m = H_px / (H_MM_REAL / 1_000)        # 2 mm → 0.002 m
    print(f"[INFO] Calibration: {px_per_m:.1f} px per metre")

    # ---- main loop ---------------------------------------------------------
    distances = []
    for i, frame in enumerate(frames):
        mask = segment_fuse(frame)
        gap_px = gap_distance(mask)
        distances.append(gap_px / px_per_m)

        if i % 20 == 0:
            print(f"  frame {i:4d}: gap = {distances[-1]*1e3:7.3f} mm")

    return distances


def save_and_plot(dists: list[float], csv_path: str = "distance.csv"):
    np.savetxt(csv_path, dists, delimiter=",")
    print(f"[INFO] Results saved to {csv_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(dists, linewidth=2)
    plt.xlabel("frame")
    plt.ylabel("distance (m)")
    plt.title("Value of d according to the frame index")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("distance_plot.png", dpi=300)
    plt.show()
    print("[INFO] Plot saved as distance_plot.png")


def main() -> None:
    argp = argparse.ArgumentParser(
        description="Measure fuse gap d on every frame of the X-ray video."
    )
    argp.add_argument("--video", required=True, help="Path to AVI/MP4 file")
    args = argp.parse_args()

    distances = process_video(args.video)
    save_and_plot(distances)


if __name__ == "__main__":
    main()
