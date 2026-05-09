# ─────────────────────────────────────────────
#  msr.py · Multi-Scale Retinex (MSR)
#  Illumination normalisation +
#  texture visibility enhancement
# ─────────────────────────────────────────────

import cv2
import numpy as np

from config import MSR_SIGMAS, MSR_GAIN, MSR_OFFSET, MSR_RESTORE_FACTOR


def _single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    SSR:  R(x,y) = log[I(x,y)] - log[I(x,y) * G(sigma)]
    where G is a Gaussian kernel of width sigma.
    Works on a single-channel float32 image.
    """
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    blur = np.where(blur < 1e-6, 1e-6, blur)   # avoid log(0)
    return np.log10(img + 1e-6) - np.log10(blur)


def apply_msr(gray: np.ndarray,
              sigmas: list = MSR_SIGMAS,
              gain: float  = MSR_GAIN,
              offset: float = MSR_OFFSET) -> np.ndarray:
    """
    Multi-Scale Retinex on a grayscale (single-channel) image.

    Algorithm:
      MSR(x,y) = Σ_s w_s * SSR_s(x,y)     (equal weights across scales)
      output   = gain * MSR + offset
      clipped to [0, 255] and returned as uint8.
    """
    img_float = gray.astype(np.float32) + 1.0   # +1 to handle zero pixels

    ssr_sum = np.zeros_like(img_float)
    weight  = 1.0 / len(sigmas)

    for sigma in sigmas:
        ssr_sum += weight * _single_scale_retinex(img_float, sigma)

    msr_out = gain * ssr_sum + offset

    msr_min, msr_max = msr_out.min(), msr_out.max()
    if msr_max - msr_min > 0:
        msr_norm = (msr_out - msr_min) / (msr_max - msr_min) * 255.0
    else:
        msr_norm = msr_out

    result = np.clip(msr_norm, 0, 255).astype(np.uint8)
    print(f"[MSR] min={result.min()}  max={result.max()}  "
          f"mean={result.mean():.2f}  sigmas={sigmas}")
    return result


def apply_msr_with_restoration(gray: np.ndarray,
                                sigmas: list = MSR_SIGMAS,
                                alpha: float = MSR_RESTORE_FACTOR) -> np.ndarray:
    """
    MSR with colour-restoration term (MSRCR) adapted for grayscale.
    The restoration factor α sharpens texture edges on low-contrast noses.
    """
    img_float = gray.astype(np.float32) + 1.0
    ssr_sum   = np.zeros_like(img_float)
    weight    = 1.0 / len(sigmas)

    for sigma in sigmas:
        ssr_sum += weight * _single_scale_retinex(img_float, sigma)

    mean_val    = img_float.mean()
    restoration = alpha * (np.log10(img_float) - np.log10(mean_val))

    enhanced = restoration * ssr_sum
    norm     = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)
