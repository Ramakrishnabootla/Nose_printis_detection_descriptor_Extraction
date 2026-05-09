# ─────────────────────────────────────────────
#  config.py · Global Configuration Constants
# ─────────────────────────────────────────────

# ── Image preprocessing settings ──
TARGET_SIZE      = (256, 256)      # resize target before MSR
GRAYSCALE        = True            # always True for ORB on nose texture

# ── Multi-Scale Retinex settings ──
MSR_SIGMAS          = [15, 80, 200]   # small / medium / large scale kernels
MSR_GAIN            = 128             # output gain
MSR_OFFSET          = 128             # output offset
MSR_RESTORE_FACTOR  = 125.0           # restoration factor α
MSR_COLOR_GAIN      = 46.0            # color gain G (unused in grayscale mode)

# ── Augmentation settings ──
AUG_ROTATION_ANGLES  = [-15, -10, -5, 5, 10, 15]  # degrees
AUG_NOISE_STD        = 8                            # Gaussian noise std-dev
AUG_BRIGHTNESS_RANGE = (0.8, 1.2)                   # multiplicative brightness jitter
AUG_FLIP_HORIZONTAL  = True

# ── ORB feature extraction settings ──
ORB_N_KEYPOINTS  = 128             # exact keypoints → (128 × 32) descriptor
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS     = 8
ORB_EDGE_THRESH  = 15
ORB_PATCH_SIZE   = 31
DESCRIPTOR_ROWS  = 128             # keypoints
DESCRIPTOR_COLS  = 32              # bytes per ORB descriptor (256 bits)

# ── Serialization settings ──
BLOB_HEADER_SIZE  = 8              # 2 × int32 header
