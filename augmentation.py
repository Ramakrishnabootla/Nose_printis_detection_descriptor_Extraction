# ─────────────────────────────────────────────
#  augmentation.py · Image Augmentation
#  rotation, flip, noise, brightness,
#  perspective warp
# ─────────────────────────────────────────────

import cv2
import numpy as np
from typing import Optional

from config import (
    AUG_ROTATION_ANGLES,
    AUG_NOISE_STD,
    AUG_BRIGHTNESS_RANGE,
    AUG_FLIP_HORIZONTAL,
)


def augment_rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by `angle` degrees around centre (no crop)."""
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def augment_flip(img: np.ndarray) -> np.ndarray:
    """Horizontal flip."""
    return cv2.flip(img, 1)


def augment_gaussian_noise(img: np.ndarray,
                            std: float = AUG_NOISE_STD) -> np.ndarray:
    """Add zero-mean Gaussian noise to simulate sensor variation."""
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def augment_brightness(img: np.ndarray,
                        factor: Optional[float] = None) -> np.ndarray:
    """
    Multiplicative brightness jitter.
    factor=None → random pick from AUG_BRIGHTNESS_RANGE.
    """
    if factor is None:
        lo, hi = AUG_BRIGHTNESS_RANGE
        factor = np.random.uniform(lo, hi)
    out = np.clip(img.astype(np.float32) * factor, 0, 255)
    return out.astype(np.uint8)


def augment_perspective(img: np.ndarray,
                         max_shift: int = 10) -> np.ndarray:
    """
    Random perspective warp — simulates slight camera angle changes
    when photographing a dog's nose.
    """
    h, w = img.shape[:2]

    def shift():
        return np.random.randint(-max_shift, max_shift)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [shift(), shift()],
        [w + shift(), shift()],
        [w + shift(), h + shift()],
        [shift(), h + shift()],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT_101)


def generate_augmentations(img: np.ndarray,
                            include_original: bool = True) -> list[np.ndarray]:
    """
    Produce all augmented variants of a preprocessed + MSR-enhanced image.
    Returns a list of uint8 grayscale images.
    """
    variants = []

    if include_original:
        variants.append(img.copy())

    # Rotations
    for angle in AUG_ROTATION_ANGLES:
        variants.append(augment_rotate(img, angle))

    # Horizontal flip
    if AUG_FLIP_HORIZONTAL:
        variants.append(augment_flip(img))
        for angle in [-10, 10]:              # flipped + rotated
            variants.append(augment_rotate(augment_flip(img), angle))

    # Noise
    variants.append(augment_gaussian_noise(img))

    # Brightness jitter (two samples)
    variants.append(augment_brightness(img, factor=0.85))
    variants.append(augment_brightness(img, factor=1.15))

    # Perspective warp
    variants.append(augment_perspective(img))

    print(f"[Augment] Generated {len(variants)} variants")
    return variants
