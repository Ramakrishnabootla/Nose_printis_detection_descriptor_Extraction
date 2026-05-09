# ─────────────────────────────────────────────
#  matcher.py · Binary Descriptor Inspection
#              & Hamming-Distance Matching
#
#  ORB descriptors are BINARY — each uint8 byte
#  packs 8 bits. The correct distance metric is
#  Hamming (count of differing bits), NOT Euclidean.
# ─────────────────────────────────────────────

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from config import DESCRIPTOR_ROWS, DESCRIPTOR_COLS


# ─────────────────────────────────────────────
#  1. Viewing the actual binary content
# ─────────────────────────────────────────────

def show_binary_representation(descriptor: np.ndarray,
                                n_keypoints: int = 4) -> None:
    """
    Print the first `n_keypoints` rows of a descriptor
    in their true binary (bit) form.

    ORB stores 256 bits per keypoint as 32 uint8 bytes.
    numpy shows them as decimal — this reveals the real bits.

    Example output for one keypoint row:
      KP[0] decimal : [232  41  17   6 ...]
      KP[0] binary  : 11101000 00101001 00010001 00000110 ...
    """
    assert descriptor.dtype == np.uint8, "Expected uint8 descriptor"
    n = min(n_keypoints, descriptor.shape[0])

    print(f"\n{'─'*60}")
    print(f"  Descriptor shape : {descriptor.shape}  "
          f"({descriptor.shape[0]} keypoints × {descriptor.shape[1]*8} bits each)")
    print(f"{'─'*60}")

    for i in range(n):
        row = descriptor[i]                        # 32 uint8 values
        bits = np.unpackbits(row)                  # 256 actual bits

        dec_str = " ".join(f"{v:3d}" for v in row[:8]) + " ..."
        bit_str = " ".join(
            "".join(str(b) for b in bits[j*8:(j+1)*8])
            for j in range(8)
        ) + " ..."

        print(f"  KP[{i}] binary  : {bit_str}")
        print()


def descriptor_to_bitmatrix(descriptor: np.ndarray) -> np.ndarray:
    """
    Expand (128, 32) uint8 → (128, 256) bool/bit matrix.
    Each row is now 256 individual bits — the true binary descriptor.
    Useful for custom Hamming implementations or visualisation.
    """
    return np.unpackbits(descriptor, axis=1)   # shape: (128, 256)


def descriptor_to_packed_bytes(bitmatrix: np.ndarray) -> np.ndarray:
    """
    Inverse of descriptor_to_bitmatrix().
    Pack (128, 256) bit matrix → (128, 32) uint8.
    """
    return np.packbits(bitmatrix, axis=1)


# ─────────────────────────────────────────────
#  2. Hamming distance helpers
# ─────────────────────────────────────────────

def hamming_distance_pair(desc_a: np.ndarray,
                           desc_b: np.ndarray) -> float:
    """
    Total Hamming distance between two (128, 32) uint8 descriptors.

    Uses numpy's bitwise XOR + popcount approach — equivalent to
    what BFMatcher NORM_HAMMING does internally.

    Lower = more similar.
    A score of 0 means identical descriptors.
    """
    assert desc_a.shape == desc_b.shape == (DESCRIPTOR_ROWS, DESCRIPTOR_COLS), \
        f"Shape mismatch: {desc_a.shape} vs {desc_b.shape}"

    # XOR → differing bits → count them
    xor = np.bitwise_xor(desc_a, desc_b)                # (128, 32)
    bit_matrix = np.unpackbits(xor, axis=1)             # (128, 256)
    return float(bit_matrix.sum())                       # total differing bits


def hamming_distance_per_keypoint(desc_a: np.ndarray,
                                   desc_b: np.ndarray) -> np.ndarray:
    """
    Per-keypoint Hamming distances between two (128, 32) descriptors.
    Returns shape (128,) — one distance per keypoint row.
    """
    xor = np.bitwise_xor(desc_a, desc_b)
    bits = np.unpackbits(xor, axis=1)                   # (128, 256)
    return bits.sum(axis=1).astype(np.float32)          # (128,)


# ─────────────────────────────────────────────
#  3. OpenCV BFMatcher — correct & fast
# ─────────────────────────────────────────────

@dataclass
class MatchResult:
    """Result of matching two nose descriptors."""
    score:           float   # lower = more similar
    good_matches:    int     # matches passing ratio test
    total_matches:   int     # raw matches before filtering
    similarity_pct:  float   # 0–100 % (100 = identical)
    is_same_dog:     bool    # threshold-based verdict


def match_descriptors(
        desc_a: np.ndarray,
        desc_b: np.ndarray,
        ratio_threshold: float = 0.75,
        similarity_threshold: float = 30.0,   # % — tune on your dataset
) -> MatchResult:
    """
    Match two (128, 32) uint8 ORB descriptors using BFMatcher
    with NORM_HAMMING + Lowe's ratio test.

    Parameters
    ----------
    desc_a / desc_b      : (128, 32) uint8 descriptors
    ratio_threshold      : Lowe ratio test cutoff (0.75 is standard)
    similarity_threshold : minimum similarity % to call a match

    Returns
    -------
    MatchResult with score, good_matches count, and verdict.

    Why NORM_HAMMING?
    -----------------
    ORB is a BINARY descriptor — each bit is independent.
    Euclidean distance (L2) is WRONG for bits.
    Hamming counts differing bits → the correct semantic distance.
    """
    # BFMatcher with Hamming norm — the only correct choice for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # knnMatch returns 2 nearest neighbours per descriptor row
    raw_matches = bf.knnMatch(desc_a, desc_b, k=2)

    # Lowe's ratio test: keep only unambiguous matches
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good.append(m)

    # Aggregate score = mean Hamming distance of good matches
    if good:
        score = float(np.mean([m.distance for m in good]))
    else:
        score = 256.0   # max possible (all 256 bits different)

    # Convert score to similarity %
    # Hamming distance per keypoint ranges [0, 256]
    # 0 → identical (100%), 256 → completely different (0%)
    similarity_pct = max(0.0, (1.0 - score / 256.0) * 100.0)

    return MatchResult(
        score           = score,
        good_matches    = len(good),
        total_matches   = len(raw_matches),
        similarity_pct  = round(similarity_pct, 2),
        is_same_dog     = similarity_pct >= similarity_threshold,
    )


# ─────────────────────────────────────────────
#  4. Batch matching (1-to-N database lookup)
# ─────────────────────────────────────────────

@dataclass
class BatchMatch:
    """Single candidate result from a 1-to-N search."""
    db_id:          int
    score:          float
    good_matches:   int
    similarity_pct: float


def find_best_match(
        query_desc: np.ndarray,
        db_descriptors: List[Tuple[int, np.ndarray]],   # [(id, descriptor), ...]
        ratio_threshold: float = 0.75,
        top_k: int = 5,
) -> List[BatchMatch]:
    """
    Compare query descriptor against a list of DB descriptors.
    Returns top-k best matches sorted by similarity (highest first).

    Usage
    -----
    db = [(1, desc_dog1), (2, desc_dog2), ...]   # loaded from PostgreSQL
    results = find_best_match(query_desc, db, top_k=3)
    for r in results:
        print(r.db_id, r.similarity_pct)
    """
    results = []
    for db_id, db_desc in db_descriptors:
        try:
            match = match_descriptors(query_desc, db_desc,
                                      ratio_threshold=ratio_threshold)
            results.append(BatchMatch(
                db_id          = db_id,
                score          = match.score,
                good_matches   = match.good_matches,
                similarity_pct = match.similarity_pct,
            ))
        except Exception as e:
            print(f"[Matcher] Skipped db_id={db_id}: {e}")

    # Sort best → worst (highest similarity first)
    results.sort(key=lambda r: r.similarity_pct, reverse=True)
    return results[:top_k]


# ─────────────────────────────────────────────
#  5. Quick demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pipeline  import run_pipeline

    if len(sys.argv) < 2:
        print("Usage: python matcher.py <image1> [image2]")
        sys.exit(1)

    # Load first image
    rec_a = run_pipeline(sys.argv[1])
    show_binary_representation(rec_a.descriptor, n_keypoints=3)

    if len(sys.argv) >= 3:
        # Match two nose images
        rec_b = run_pipeline(sys.argv[2])
        result = match_descriptors(rec_a.descriptor, rec_b.descriptor)

        print(f"\n{'─'*50}")
        print(f"  MATCH RESULT")
        print(f"{'─'*50}")
        print(f"  Score (mean Hamming) : {result.score:.2f}  / 256 max")
        print(f"  Good matches         : {result.good_matches} / {result.total_matches}")
        print(f"  Similarity           : {result.similarity_pct} %")
        print(f"  Verdict              : {'✓ SAME DOG' if result.is_same_dog else '✗ DIFFERENT DOG'}")
    else:
        # Self-match sanity check (should be ~100%)
        result = match_descriptors(rec_a.descriptor, rec_a.descriptor)
        print(f"\n[Self-match sanity check]")
        print(f"  Similarity : {result.similarity_pct} %  (expect ~100)")