# ─────────────────────────────────────────────
#  main.py · Entry Point
#  Run the complete dog nose-print pipeline
#  from the command line.
#
#  Usage:
#    python main.py <image_path> [options]
#
#  Options:
#    --restore     Use MSRCR (restoration variant of MSR)
#    --augment     Generate augmented variants and process each
#    --preview     Save keypoint visualisation image to disk
#
#  Example:
#    python main.py dog_nose.jpg --augment --preview
# ─────────────────────────────────────────────

import sys
import argparse
import numpy as np
import json

from pipeline        import run_pipeline, NosePrintRecord
from orb_extractor   import visualize_keypoints
from serializer      import descriptor_to_hex
from matcher         import show_binary_representation, descriptor_to_bitmatrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dog Nose-Print Biometric Pipeline"
    )
    parser.add_argument(
        "image_path",
        help="Path to the input nose image (jpg/png/bmp/tiff)"
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        default=False,
        help="Use MSRCR restoration variant instead of standard MSR"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Generate augmented variants and extract descriptors from each"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help="Save a keypoint visualisation image alongside the input"
    )
    return parser.parse_args()


def print_record_summary(record: NosePrintRecord, index: int | None = None) -> None:
    """Pretty-print a single NosePrintRecord to the terminal."""
    label = f"Record [{index}]" if index is not None else "Record"
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"  image      : {record.image_path}")
    print(f"  sha256     : {record.image_hash[:24]}…")
    print(f"  keypoints  : {record.keypoint_count}")
    print(f"  desc shape : {record.descriptor.shape}")
    print(f"  blob bytes : {len(record.descriptor_blob)}")
    print(f"  hex preview: {descriptor_to_hex(record.descriptor)[:32]}…")


def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("  Dog Nose-Print Biometric Pipeline")
    print("=" * 50)
    print(f"  Image        : {args.image_path}")
    print(f"  MSR variant  : {'MSRCR (restoration)' if args.restore else 'Standard MSR'}")
    print(f"  Augmentation : {'Yes' if args.augment else 'No'}")
    print(f"  KP preview   : {'Yes' if args.preview else 'No'}")
    print("=" * 50)

    # ── Run pipeline ─────────────────────────────────────────────────────────
    result = run_pipeline(
        image_path          = args.image_path,
        use_restoration_msr = args.restore,
        augment             = args.augment,
    )

    # ── Display results ───────────────────────────────────────────────────────
    if args.augment:
        records: list[NosePrintRecord] = result
        print(f"\n[Main] {len(records)} records produced from augmentation.")
        for i, rec in enumerate(records):
            print_record_summary(rec, index=i)

        # Show descriptor preview for the first record
        if records:
            print("\n── Descriptor preview (all rows of record 0) ──")
            print(records[0].descriptor)

            # Show binary representation
            show_binary_representation(records[0].descriptor)

            # Save descriptor in binary format as array of arrays
            bitmatrix = descriptor_to_bitmatrix(records[0].descriptor)
            binary_filename = args.image_path.rsplit(".", 1)[0] + "_descriptor.json"
            bitmatrix_list = bitmatrix.astype(int).tolist()  # Convert bool to int for JSON
            with open(binary_filename, 'w') as f:
                f.write('[\n')
                for i, row in enumerate(bitmatrix_list):
                    f.write('  ' + json.dumps(row))
                    if i < len(bitmatrix_list) - 1:
                        f.write(',\n')
                    else:
                        f.write('\n')
                f.write(']\n')
            print(f"\n[Main] Descriptor saved in JSON format (128x256 bit array) → {binary_filename}")

        # Save keypoint visualisation for the original (record 0)
        if args.preview and records:
            import cv2
            from image_input   import load_image
            from preprocessing import preprocess_image
            from msr           import apply_msr, apply_msr_with_restoration

            raw   = load_image(args.image_path)
            pre   = preprocess_image(raw)
            enh   = apply_msr_with_restoration(pre) if args.restore else apply_msr(pre)
            save  = args.image_path.rsplit(".", 1)[0] + "_keypoints.jpg"
            visualize_keypoints(enh, records[0].keypoints, save_path=save)
            print(f"\n[Main] Keypoint visualisation saved → {save}")

    else:
        record: NosePrintRecord = result
        print_record_summary(record)

        print("\n── Descriptor preview (all rows) ──")
        # print(record.descriptor)

        # Show binary representation
        show_binary_representation(record.descriptor)

        # Save descriptor in binary format as array of arrays
        bitmatrix = descriptor_to_bitmatrix(record.descriptor)
        binary_filename = args.image_path.rsplit(".", 1)[0] + "_descriptor.json"
        bitmatrix_list = bitmatrix.astype(int).tolist()  # Convert bool to int for JSON
        with open(binary_filename, 'w') as f:
            f.write('[\n')
            for i, row in enumerate(bitmatrix_list):
                f.write('  ' + json.dumps(row))
                if i < len(bitmatrix_list) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            f.write(']\n')
        print(f"\n[Main] Descriptor saved in JSON format (128x256 bit array) → {binary_filename}")

        # Optionally save keypoint visualisation
        if args.preview:
            import cv2
            from image_input   import load_image
            from preprocessing import preprocess_image
            from msr           import apply_msr, apply_msr_with_restoration

            raw  = load_image(args.image_path)
            pre  = preprocess_image(raw)
            enh  = apply_msr_with_restoration(pre) if args.restore else apply_msr(pre)
            save = args.image_path.rsplit(".", 1)[0] + "_keypoints.jpg"
            visualize_keypoints(enh, record.keypoints, save_path=save)
            print(f"\n[Main] Keypoint visualisation saved → {save}")

    print("\n[Main] Pipeline finished successfully.")


if __name__ == "__main__":
    main()
