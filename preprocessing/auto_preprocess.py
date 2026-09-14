"""
Quality-Aware OCR Image Preprocessing Pipeline CLI.

Provides automated single-image and batch-directory processing for document pages,
applying non-destructive transformations only when flagged by the quality gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from preprocessing.quality_gate import assess_image_file, assess_page
from preprocessing.kannada_preprocess import preprocess_page

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def process_one(
    src: str | Path,
    dst: str | Path,
    verbose: bool = True,
) -> dict:
    """
    Process a single document image based on quality gate evaluation.

    Good pages are kept unchanged.
    Degraded pages receive only the operations selected by the quality gate.
    """
    input_path = Path(src)
    output_path = Path(dst)

    image = cv2.imread(str(input_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    # Step 1: assess page quality
    if verbose:
        quality = assess_image_file(str(input_path))
    else:
        quality = assess_page(image)

    # Step 2: decide action based on operations
    if not quality.operations:
        processed = image.copy()
        if verbose:
            print()
            print("========== AUTO PREPROCESS ==========")
            print(f"Input      : {input_path}")
            print("Decision   : KEEP ORIGINAL")
            print("Operations : NONE")
            print(f"Output     : {output_path}")
            print("=====================================")
    else:
        processed = preprocess_page(
            image,
            skew_angle=quality.skew_angle,
            operations=quality.operations,
        )
        if verbose:
            print()
            print("========== AUTO PREPROCESS ==========")
            print(f"Input      : {input_path}")
            print("Decision   : PREPROCESS")
            print(f"Operations : {quality.operations}")
            print(f"Output     : {output_path}")
            print("=====================================")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(str(output_path), processed):
        raise RuntimeError(f"Could not write output: {output_path}")

    return {
        "src": str(input_path),
        "dst": str(output_path),
        "preprocessed": bool(quality.operations),
        "operations": quality.operations,
        "quality": quality,
    }


def auto_preprocess(
    input_path: str | Path,
    output_path: str | Path,
    verbose: bool = True,
) -> dict:
    """Backward-compatible wrapper around process_one."""
    return process_one(input_path, output_path, verbose=verbose)


def process_batch(
    src_dir: str | Path,
    dst_dir: str | Path,
    recursive: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Process every supported image in src_dir and write outputs to dst_dir,
    preserving relative subdirectory paths.
    """
    source_path = Path(src_dir)
    target_path = Path(dst_dir)

    if not source_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {source_path}")

    # Collect files
    if recursive:
        candidates = source_path.rglob("*")
    else:
        candidates = source_path.glob("*")

    image_files = sorted(
        [p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    results: list[dict] = []

    for img_path in image_files:
        rel_path = img_path.relative_to(source_path)
        out_file = target_path / rel_path

        try:
            res = process_one(img_path, out_file, verbose=verbose)
            results.append(res)
        except Exception as exc:
            print(f"Error processing {img_path}: {exc}", file=sys.stderr)
            results.append(
                {
                    "src": str(img_path),
                    "dst": str(out_file),
                    "preprocessed": False,
                    "operations": [],
                    "error": str(exc),
                }
            )

    total = len(image_files)
    errors = sum(1 for r in results if "error" in r)
    preprocessed = sum(1 for r in results if r.get("preprocessed") and "error" not in r)
    kept_original = sum(1 for r in results if not r.get("preprocessed") and "error" not in r)
    processed_count = total - errors

    if verbose:
        print()
    print("========== BATCH SUMMARY ==========")
    print(f"Processed : {processed_count}/{total}")
    print(f"Preprocessed: {preprocessed}")
    print(f"Kept original: {kept_original}")
    print(f"Errors    : {errors}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quality-aware OCR image preprocessing pipeline"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input image file or directory",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to output image file or directory",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively process subdirectories",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-page diagnostic output",
    )

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    verbose = not args.quiet

    if input_path.is_dir():
        process_batch(
            src_dir=input_path,
            dst_dir=output_path,
            recursive=args.recursive,
            verbose=verbose,
        )
    elif input_path.is_file():
        # If output is a directory or has no extension, route to output / input.name
        if output_path.is_dir() or not output_path.suffix:
            output_path = output_path / input_path.name

        process_one(
            src=input_path,
            dst=output_path,
            verbose=verbose,
        )
    else:
        print(f"Error: Input path '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
