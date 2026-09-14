"""
Kannada Document OCR Handoff & Preparation Pipeline.

This module serves as the complete handoff stage prior to OCR ingestion. It accepts
either a multi-page PDF document or a single image, evaluates each page against the
quality gate, selectively applies non-destructive preprocessing transformations, and
writes normalized grayscale page images along with an index manifest to the output
directory.

Downstream consumers (such as Tesseract, PaddleOCR, or custom OCR engines) read
the generated output directory and its manifest JSON file. This module does NOT
perform any OCR recognition.

Error Policy:
Unreadable or corrupt pages raise a RuntimeError immediately, preventing partial or
corrupted document outputs and maintaining data integrity for downstream consumers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

try:
    from .kannada_preprocess import preprocess_page
    from .quality_gate import assess_page
except ImportError:
    from preprocessing.kannada_preprocess import preprocess_page
    from preprocessing.quality_gate import assess_page

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pnm"}


@dataclass
class PageManifest:
    page_index: int
    source: str
    output: str
    width: int
    height: int
    decision: str  # "KEEP" | "PREPROCESS"
    operations: list[str]
    metrics: dict  # keys: skew_deg, blur, contrast, illumination, noise


@dataclass
class DocManifest:
    source: str
    pages: list[PageManifest]

    @property
    def n_preprocessed(self) -> int:
        return sum(1 for p in self.pages if p.decision == "PREPROCESS")

    @property
    def n_kept(self) -> int:
        return sum(1 for p in self.pages if p.decision == "KEEP")


def prepare_page(
    image_path: str | Path,
    output_path: str | Path,
    page_index: int,
    force_ops: list[str] | None = None,
    source_label: str | None = None,
) -> PageManifest:
    """
    Assess and prepare a single page for downstream OCR consumption.

    Loads grayscale image, determines needed operations via the quality gate
    (or force_ops override), conditionally executes preprocessing, writes the
    output image, and returns the page manifest.
    """
    src = Path(image_path)
    dst = Path(output_path)

    image = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {src}")

    q = assess_page(image)
    ops = list(force_ops) if force_ops is not None else list(q.operations)

    if ops:
        processed = preprocess_page(
            image,
            skew_angle=q.skew_angle,
            operations=ops,
        )
        decision = "PREPROCESS"
    else:
        processed = image
        decision = "KEEP"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), processed):
        raise RuntimeError(f"Could not write output image: {dst}")

    height, width = image.shape[:2]
    metrics = {
        "skew_deg": float(q.skew_angle),
        "blur": float(q.blur_score),
        "contrast": float(q.contrast_score),
        "illumination": float(q.illumination_score),
        "noise": float(q.noise_score),
    }

    return PageManifest(
        page_index=page_index,
        source=str(source_label or src),
        output=dst.name,
        width=width,
        height=height,
        decision=decision,
        operations=ops,
        metrics=metrics,
    )


def prepare_document(
    src: str | Path,
    out_dir: str | Path,
    dpi: int = 300,
    force_ops: list[str] | None = None,
    verbose: bool = True,
) -> DocManifest:
    """
    Process an entire PDF or single image into OCR-ready images and a JSON manifest.
    """
    src_path = Path(src)
    output_dir = Path(out_dir)

    if not src_path.is_file():
        raise FileNotFoundError(f"Input file not found: {src_path}")

    ext = src_path.suffix.lower()
    if ext not in PDF_EXTENSIONS and ext not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: "
            f"{sorted(PDF_EXTENSIONS | IMAGE_EXTENSIONS)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = src_path.stem
    manifest_filename = f"{stem}_manifest.json"
    manifest_path = output_dir / manifest_filename

    temp_dir = None
    try:
        if ext in PDF_EXTENSIONS:
            if shutil.which("pdftoppm") is None:
                raise RuntimeError(
                    "pdftoppm not found. Please install poppler-utils "
                    "(e.g. 'sudo apt install poppler-utils') to rasterize PDF documents."
                )

            temp_dir = tempfile.mkdtemp(prefix="ocr_pdf_")
            temp_prefix = Path(temp_dir) / "page"

            cmd = [
                "pdftoppm",
                "-png",
                "-r",
                str(dpi),
                str(src_path),
                str(temp_prefix),
            ]
            run_res = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if run_res.returncode != 0:
                raise RuntimeError(
                    f"pdftoppm failed with code {run_res.returncode}: {run_res.stderr.strip()}"
                )

            page_files = sorted(
                Path(temp_dir).glob("*.png"),
                key=lambda p: int(p.stem.split("-")[-1])
                if p.stem.split("-")[-1].isdigit()
                else p.stem,
            )

            if not page_files:
                raise RuntimeError(f"pdftoppm yielded no pages for: {src_path}")
        else:
            page_files = [src_path]

        total_pages = len(page_files)
        pages: list[PageManifest] = []

        w = max(len(str(total_pages)), 3)

        for idx, page_file in enumerate(page_files, start=1):
            page_filename = f"{stem}_page-{idx:04d}.png"
            page_out_path = output_dir / page_filename

            manifest_entry = prepare_page(
                image_path=page_file,
                output_path=page_out_path,
                page_index=idx,
                force_ops=force_ops,
                source_label=str(src_path),
            )
            pages.append(manifest_entry)

            if verbose:
                ops_str = (
                    f"ops={','.join(manifest_entry.operations)}"
                    if manifest_entry.operations
                    else "ops=none"
                )
                print(
                    f"[{idx:{w}d}/{total_pages:{w}d}] "
                    f"{manifest_entry.decision:<10}  "
                    f"{ops_str:<43} -> "
                    f"{page_filename}"
                )

        doc_manifest = DocManifest(
            source=str(src_path),
            pages=pages,
        )

        manifest_json = json.dumps(
            asdict(doc_manifest),
            indent=2,
            ensure_ascii=False,
        )
        manifest_path.write_text(manifest_json, encoding="utf-8")

        if verbose:
            out_display = str(output_dir).rstrip("/") + "/"
            print(
                f"wrote {total_pages} page(s) + {manifest_filename} to {out_display}"
            )
            print(f"  preprocessed: {doc_manifest.n_preprocessed}")
            print(f"  kept original: {doc_manifest.n_kept}")

        return doc_manifest

    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare PDF or image documents for downstream OCR"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input PDF or image file",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory to write prepared pages and manifest",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization resolution for PDF files (default: 300)",
    )
    parser.add_argument(
        "--force-ops",
        type=str,
        default=None,
        help="Comma-separated operations to force (empty string forces KEEP)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-page and summary console output",
    )

    args = parser.parse_args()

    if args.force_ops is not None:
        force_ops = [x.strip() for x in args.force_ops.split(",") if x.strip()]
    else:
        force_ops = None

    prepare_document(
        src=args.input,
        out_dir=args.output,
        dpi=args.dpi,
        force_ops=force_ops,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
