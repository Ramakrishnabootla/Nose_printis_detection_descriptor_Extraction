# ─────────────────────────────────────────────
#  image_input.py · Image Input
#  Handles loading a single dog nose image
#  from disk and computing its SHA-256 hash.
# ─────────────────────────────────────────────

import cv2
import hashlib
import numpy as np
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_image(image_path: str) -> np.ndarray:
    """
    Load a single dog nose image from disk.
    Returns the raw BGR image (uint8).
    Raises FileNotFoundError or ValueError on failure.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {ext}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    print(f"[Input] Loaded '{path.name}'  shape={img.shape}  dtype={img.dtype}")
    return img


def get_image_hash(image_path: str) -> str:
    """SHA-256 hash of the raw file bytes — used as dedup key in PostgreSQL."""
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
