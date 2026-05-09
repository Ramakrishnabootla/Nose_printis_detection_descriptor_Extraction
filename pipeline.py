# ─────────────────────────────────────────────
#  pipeline.py · Full Pipeline Runner
#  Ties all modules together for a single
#  input image and returns a NosePrintRecord.
# ─────────────────────────────────────────────

from dataclasses import dataclass

import numpy as np

from image_input   import load_image, get_image_hash
from preprocessing import preprocess_image
from msr           import apply_msr, apply_msr_with_restoration
from augmentation  import generate_augmentations
from orb_extractor import build_orb_detector, extract_orb_descriptors
from serializer    import serialize_descriptors, validate_blob


@dataclass
class NosePrintRecord:
    """Structured output ready to be inserted into PostgreSQL."""
    image_path:      str
    image_hash:      str
    descriptor:      np.ndarray   # shape (128, 32), dtype uint8
    descriptor_blob: bytes        # serialized BYTEA
    keypoint_count:  int
    keypoints:       list         # list of cv2.KeyPoint (not stored in DB)


def run_pipeline(
        image_path: str,
        use_restoration_msr: bool = False,
        augment: bool = False
) -> "NosePrintRecord | list[NosePrintRecord]":
    """
    End-to-end pipeline for a single nose image.

    Parameters
    ----------
    image_path           : path to input image
    use_restoration_msr  : if True, uses MSRCR (restoration variant)
    augment              : if True, returns a list of NosePrintRecord
                           (one per augmented variant)

    Returns
    -------
    NosePrintRecord            when augment=False
    list[NosePrintRecord]      when augment=True
    """

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    raw_img  = load_image(image_path)
    img_hash = get_image_hash(image_path)

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    preprocessed = preprocess_image(raw_img)     # resize → gray → CLAHE

    # ── Step 3: MSR Enhancement ──────────────────────────────────────────────
    if use_restoration_msr:
        enhanced = apply_msr_with_restoration(preprocessed)
    else:
        enhanced = apply_msr(preprocessed)

    # ── Step 4: Build ORB detector (reused across augmentations) ─────────────
    orb = build_orb_detector()

    def _extract_one(img: np.ndarray) -> NosePrintRecord:
        kps, desc = extract_orb_descriptors(img, orb)
        blob      = serialize_descriptors(desc)
        assert validate_blob(blob), "Descriptor blob failed validation!"
        return NosePrintRecord(
            image_path      = image_path,
            image_hash      = img_hash,
            descriptor      = desc,
            descriptor_blob = blob,
            keypoint_count  = len(kps),
            keypoints       = kps,
        )

    # ── Step 5 (optional): Augmentation ──────────────────────────────────────
    if augment:
        variants = generate_augmentations(enhanced, include_original=True)
        records  = []
        for i, aug_img in enumerate(variants):
            try:
                rec = _extract_one(aug_img)
                records.append(rec)
                print(f"[Pipeline] Augment {i+1}/{len(variants)}  "
                      f"kp={rec.keypoint_count}")
            except ValueError as e:
                print(f"[Pipeline] Augment {i+1} skipped: {e}")
        return records

    # ── Step 5: Single record ─────────────────────────────────────────────────
    record = _extract_one(enhanced)
    print(f"\n[Pipeline] ✓ Complete"
          f"\n  image      : {image_path}"
          f"\n  sha256     : {img_hash[:16]}…"
          f"\n  keypoints  : {record.keypoint_count}"
          f"\n  desc shape : {record.descriptor.shape}"
          f"\n  blob bytes : {len(record.descriptor_blob)}")
    return record
