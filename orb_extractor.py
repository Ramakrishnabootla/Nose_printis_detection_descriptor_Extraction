# ─────────────────────────────────────────────
#  orb_extractor.py · ORB Feature Extraction
#  Output: (128, 32) uint8 descriptor
#  = 128 keypoints × 256-bit binary
# ─────────────────────────────────────────────

import cv2
import numpy as np
from typing import Optional, Tuple

from config import (
    ORB_N_KEYPOINTS,
    ORB_SCALE_FACTOR,
    ORB_N_LEVELS,
    ORB_EDGE_THRESH,
    ORB_PATCH_SIZE,
    DESCRIPTOR_ROWS,
    DESCRIPTOR_COLS,
)


def build_orb_detector(n_keypoints: int = ORB_N_KEYPOINTS) -> cv2.ORB:
    """
    Construct ORB detector with settings tuned for nose-texture matching.
    nfeatures=128 → exactly 128 keypoints → (128, 32) descriptor matrix.
    """
    return cv2.ORB_create(
        nfeatures     = n_keypoints,
        scaleFactor   = ORB_SCALE_FACTOR,
        nlevels       = ORB_N_LEVELS,
        edgeThreshold = ORB_EDGE_THRESH,
        firstLevel    = 0,
        WTA_K         = 2,               # 2 → NORM_HAMMING
        scoreType     = cv2.ORB_HARRIS_SCORE,
        patchSize     = ORB_PATCH_SIZE,
        fastThreshold = 10,
    )


def pad_or_trim_descriptors(descriptors: np.ndarray,
                             target_rows: int = DESCRIPTOR_ROWS) -> np.ndarray:
    """
    Enforce a fixed (128, 32) shape regardless of how many keypoints ORB found.

    - Fewer than 128 → zero-pad remaining rows.
    - More than 128  → keep highest-scored (first rows, ORB returns them
      sorted by Harris score descending).
    """
    rows, cols = descriptors.shape
    if rows >= target_rows:
        return descriptors[:target_rows, :]

    pad = np.zeros((target_rows - rows, cols), dtype=np.uint8)
    return np.vstack([descriptors, pad])


def extract_orb_descriptors(
        img: np.ndarray,
        orb: Optional[cv2.ORB] = None
) -> Tuple[list, np.ndarray]:
    """
    Run ORB on a preprocessed, MSR-enhanced grayscale image.

    Returns
    -------
    keypoints   : list of cv2.KeyPoint
    descriptors : np.ndarray  shape=(128, 32)  dtype=uint8
                  Always exactly 128 rows — padded with zeros if needed.

    Raises
    ------
    ValueError if no keypoints at all are detected.
    """
    if orb is None:
        orb = build_orb_detector()

    keypoints, raw_desc = orb.detectAndCompute(img, None)

    if raw_desc is None or len(keypoints) == 0:
        raise ValueError(
            "ORB found 0 keypoints. "
            "Check that the image contains visible nose texture."
        )

    descriptors = pad_or_trim_descriptors(raw_desc, DESCRIPTOR_ROWS)

    actual_kp = min(len(keypoints), DESCRIPTOR_ROWS)
    print(f"[ORB] Keypoints detected={len(keypoints)}  "
          f"Used={actual_kp}  "
          f"Descriptor shape={descriptors.shape}  "
          f"dtype={descriptors.dtype}")

    return keypoints, descriptors


def visualize_keypoints(img: np.ndarray,
                         keypoints: list,
                         save_path: Optional[str] = None) -> np.ndarray:
    """Draw keypoints on the image. Optionally save to disk."""
    vis = cv2.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    if save_path:
        cv2.imwrite(save_path, vis)
    return vis
