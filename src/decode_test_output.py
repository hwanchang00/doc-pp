"""
Decode test output JSON file and save images/PDFs to files.

Usage:
    python decode_test_output.py <json_file>

Example:
    python decode_test_output.py data/test_evaluation_output.json
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List


def decode_and_save(json_path: Path, output_dir: Path) -> None:
    """
    Read the JSON file and dump embedded base64 images/PDFs to disk.

    Args:
        json_path: Path to the input JSON file
        output_dir: Directory where decoded assets should be stored
    """
    # Load the JSON payload
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data).__name__}")

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(data)} items from {json_path}")
    print(f"Output directory: {output_dir}")
    print()

    for idx, item in enumerate(data):
        instance_id = item.get("id", idx)
        doc_id = item.get("doc_id", f"unknown_{idx}")
        test_info = item.get("test_info", {})
        doc_mode = test_info.get("doc_mode", "unknown")

        print(f"[{idx+1}/{len(data)}] ID={instance_id}, doc_id={doc_id}, mode={doc_mode}")

        # PDF mode
        if doc_mode == "pdf":
            pdf_base64 = test_info.get("pdf_base64", "")
            if not pdf_base64:
                print(f"  ⚠ No PDF data found")
                continue

            # Persist the PDF
            pdf_filename = f"instance_{instance_id}_{doc_id}"
            pdf_path = output_dir / pdf_filename

            try:
                pdf_bytes = base64.b64decode(pdf_base64)
                pdf_path.write_bytes(pdf_bytes)
                print(f"  ✓ Saved PDF: {pdf_path} ({len(pdf_bytes)} bytes)")
            except Exception as e:
                print(f"  ✗ Failed to decode PDF: {e}")

        # Image mode
        elif doc_mode == "image":
            images_base64 = test_info.get("images_base64", [])
            num_images = len(images_base64)

            if num_images == 0:
                print(f"  ⚠ No images found")
                continue

            print(f"  Processing {num_images} images...")

            for img_idx, img_base64 in enumerate(images_base64):
                img_filename = f"instance_{instance_id}_{doc_id.replace('.pdf', '')}_page{img_idx+1}.png"
                img_path = output_dir / img_filename

                try:
                    img_bytes = base64.b64decode(img_base64)
                    img_path.write_bytes(img_bytes)
                    print(f"    ✓ Saved image {img_idx+1}: {img_path} ({len(img_bytes)} bytes)")
                except Exception as e:
                    print(f"    ✗ Failed to decode image {img_idx+1}: {e}")
        else:
            print(f"  ⚠ Unknown doc_mode: {doc_mode}")

        print()

    print(f"Done! Check {output_dir} for saved files.")


def main():
    parser = argparse.ArgumentParser(
        description="Decode test output JSON and save images/PDFs to files."
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to test output JSON file (e.g., data/test_evaluation_output.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for decoded files. Default: <json_file_dir>/decoded_<json_filename>",
    )

    args = parser.parse_args()

    json_path = Path(args.json_file)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Choose output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: create decoded_<filename> next to the JSON
        output_dir = json_path.parent / f"decoded_{json_path.stem}"

    decode_and_save(json_path, output_dir)


if __name__ == "__main__":
    main()
