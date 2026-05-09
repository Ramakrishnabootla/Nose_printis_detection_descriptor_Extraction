# ─────────────────────────────────────────────
#  preprocessing.py · Image Preprocessing
#  Pipeline: resize → grayscale → CLAHE
# ─────────────────────────────────────────────

import cv2
import numpy as np
from typing import Tuple

from config import TARGET_SIZE


def resize_image(img: np.ndarray,
                 target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Resize to a fixed square while preserving aspect ratio with padding.
    Padding colour is black (0).
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Centre-pad to exact target size
    canvas = np.zeros((target_h, target_w) + img.shape[2:], dtype=np.uint8)
    y_off  = (target_h - new_h) // 2
    x_off  = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if img.ndim == 2:
        return img   # already grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Boosts local contrast on nose texture before Retinex.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Full preprocessing chain:
      raw BGR  →  resize  →  grayscale  →  CLAHE
    Returns uint8 grayscale image of shape TARGET_SIZE.
    """
    img_resized = resize_image(img)
    img_gray    = convert_to_grayscale(img_resized)
    img_clahe   = apply_clahe(img_gray)
    print(f"[Preprocess] Output shape={img_clahe.shape}  dtype={img_clahe.dtype}")
    return img_clahe
