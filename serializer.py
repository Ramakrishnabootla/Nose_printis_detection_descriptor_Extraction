# ─────────────────────────────────────────────
#  serializer.py · Descriptor Serialization
#  BYTEA ↔ np.ndarray helpers for PostgreSQL
# ─────────────────────────────────────────────

import numpy as np

from config import DESCRIPTOR_ROWS, DESCRIPTOR_COLS, BLOB_HEADER_SIZE

# Fixed blob size: 8 bytes header + 128*32 = 4096 bytes = 4104 bytes total
BLOB_HEADER_DTYPE = np.int32


def serialize_descriptors(descriptors: np.ndarray) -> bytes:
    """
    Pack (128, 32) uint8 array → bytes for PostgreSQL BYTEA.

    Layout: [rows:int32][cols:int32][flat uint8 data...]
    Total:  8 + 128*32 = 4104 bytes per record.
    """
    assert descriptors.dtype == np.uint8, "Descriptors must be uint8"
    assert descriptors.ndim  == 2,        "Descriptors must be 2-D"
    rows, cols = descriptors.shape
    header = np.array([rows, cols], dtype=BLOB_HEADER_DTYPE).tobytes()
    return header + descriptors.tobytes()


def deserialize_descriptors(blob: bytes) -> np.ndarray:
    """
    Unpack bytes from PostgreSQL BYTEA → (128, 32) uint8 array.
    Inverse of serialize_descriptors().
    """
    rows, cols = np.frombuffer(blob[:BLOB_HEADER_SIZE], dtype=BLOB_HEADER_DTYPE)
    data = np.frombuffer(blob[BLOB_HEADER_SIZE:], dtype=np.uint8)
    return data.reshape((int(rows), int(cols)))


def descriptor_to_hex(descriptors: np.ndarray) -> str:
    """
    Convert descriptor matrix to hex string (useful for logging/debugging).
    Not stored in DB — use serialize_descriptors() for that.
    """
    return serialize_descriptors(descriptors).hex()


def validate_blob(blob: bytes) -> bool:
    """Quick sanity check before inserting into DB."""
    expected = BLOB_HEADER_SIZE + DESCRIPTOR_ROWS * DESCRIPTOR_COLS
    if len(blob) != expected:
        print(f"[Validate] FAIL: expected {expected} bytes, got {len(blob)}")
        return False
    desc = deserialize_descriptors(blob)
    ok   = (desc.shape == (DESCRIPTOR_ROWS, DESCRIPTOR_COLS)
            and desc.dtype == np.uint8)
    print(f"[Validate] {'PASS' if ok else 'FAIL'}  "
          f"shape={desc.shape}  dtype={desc.dtype}")
    return ok
