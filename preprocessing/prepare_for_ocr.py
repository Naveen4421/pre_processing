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
    decision: str  # "KEEP" | "PREPROCESS" | "PENDING"
    operations: list[str]
    metrics: dict  # keys: skew_deg, blur, contrast, illumination, noise
    ocr_text: str = ""


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


def run_page_ocr(image_path: str | Path, lang: str = "kan") -> str:
    """
    Run OCR (Tesseract) on a preprocessed page image and return recognized text.
    Returns empty string if tesseract is not found or fails.
    """
    tess = shutil.which("tesseract")
    if not tess:
        return ""
    img_p = Path(image_path)
    if not img_p.is_file():
        return ""
    with tempfile.TemporaryDirectory(prefix="ocr_tess_") as td:
        out_base = Path(td) / "out"
        cmd = [
            tess,
            str(img_p),
            str(out_base),
            "-l",
            lang,
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        out_txt = Path(td) / "out.txt"
        if out_txt.is_file():
            return out_txt.read_text(encoding="utf-8", errors="replace").strip()
    return ""


def prepare_page(
    image_path: str | Path,
    output_path: str | Path,
    page_index: int,
    force_ops: list[str] | None = None,
    source_label: str | None = None,
    run_ocr: bool = False,
    ocr_lang: str = "kan",
) -> PageManifest:
    """
    Assess and prepare a single page for downstream OCR consumption.

    Loads grayscale image, determines needed operations via the quality gate
    (or force_ops override), conditionally executes preprocessing, writes the
    output image, optionally runs OCR immediately on the result, and returns
    the page manifest.
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
            orientation_angle=q.orientation_angle,
            operations=ops,
        )
        decision = "PREPROCESS"
    else:
        processed = image
        decision = "KEEP"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), processed):
        raise RuntimeError(f"Could not write output image: {dst}")

    height, width = processed.shape[:2]
    metrics = {
        "skew_deg": float(q.skew_angle),
        "orientation_deg": int(q.orientation_angle),
        "blur": float(q.blur_score),
        "contrast": float(q.contrast_score),
        "illumination": float(q.illumination_score),
        "noise": float(q.noise_score),
    }

    ocr_text = ""
    if run_ocr:
        ocr_text = run_page_ocr(dst, lang=ocr_lang)
        txt_path = dst.with_suffix(".txt")
        txt_path.write_text(ocr_text, encoding="utf-8")

    return PageManifest(
        page_index=page_index,
        source=str(source_label or src),
        output=dst.name,
        width=width,
        height=height,
        decision=decision,
        operations=ops,
        metrics=metrics,
        ocr_text=ocr_text,
    )


def get_pdf_page_count(pdf_path: str | Path) -> int:
    """Return total page count of a PDF using pdfinfo or regex fallback."""
    pdf_p = Path(pdf_path)
    if not pdf_p.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_p}")

    if shutil.which("pdfinfo") is not None:
        try:
            res = subprocess.run(
                ["pdfinfo", str(pdf_p)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            for line in res.stdout.splitlines():
                if line.startswith("Pages:"):
                    return int(line.split(":")[1].strip())
        except Exception:
            pass

    # Fallback to scanning /Type /Page
    try:
        with open(pdf_p, "rb") as f:
            content = f.read()
            import re

            matches = re.findall(rb"/Type\s*/Page\b", content)
            if matches:
                return len(matches)
    except Exception:
        pass

    return 1


def extract_pdf_page(
    pdf_path: str | Path,
    page_index: int,
    output_png: str | Path,
    dpi: int = 300,
) -> Path:
    """
    Extract a single page from a PDF at the given DPI using pdftoppm.
    page_index is 1-indexed.
    """
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm not found. Please install poppler-utils.")

    out_p = Path(output_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"extract_p{page_index}_") as td:
        prefix = Path(td) / "page"
        cmd = [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            "-f",
            str(page_index),
            "-l",
            str(page_index),
            str(pdf_path),
            str(prefix),
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"pdftoppm failed for page {page_index}: {res.stderr.strip()}"
            )

        extracted = list(Path(td).glob("*.png"))
        if not extracted:
            raise RuntimeError(
                f"pdftoppm yielded no image for page {page_index} of {pdf_path}"
            )

        shutil.move(str(extracted[0]), str(out_p))

    return out_p


def parse_page_selection(spec: str, total_pages: int) -> list[int]:
    """Parse page specification like '1-5', '1,3,7', '42' into 1-indexed integers."""
    selected = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            for p in range(start, end + 1):
                if 1 <= p <= total_pages:
                    selected.add(p)
        else:
            p = int(part)
            if 1 <= p <= total_pages:
                selected.add(p)
    return sorted(selected)


def stream_prepare_document(
    src: str | Path,
    out_dir: str | Path,
    dpi: int = 300,
    force_ops: list[str] | None = None,
    page_numbers: list[int] | None = None,
    run_ocr: bool = False,
    ocr_lang: str = "kan",
    callback: callable | None = None,
):
    """
    Pipelined streaming processor:
    Yields each page as soon as it is preprocessed and optionally OCR'd,
    allowing downstream OCR consumers to ingest page-by-page without waiting
    for the rest of the document to finish.
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

    pages: list[PageManifest] = []

    def _sync_manifest():
        doc_manifest = DocManifest(source=str(src_path), pages=pages)
        manifest_json = json.dumps(asdict(doc_manifest), indent=2, ensure_ascii=False)
        manifest_path.write_text(manifest_json, encoding="utf-8")

    if ext in PDF_EXTENSIONS:
        if shutil.which("pdftoppm") is None:
            raise RuntimeError(
                "pdftoppm not found. Please install poppler-utils "
                "(e.g. 'sudo apt install poppler-utils') to rasterize PDF documents."
            )

        total_doc_pages = get_pdf_page_count(src_path)
        if page_numbers:
            target_indices = sorted(set(p for p in page_numbers if 1 <= p <= total_doc_pages))
            if not target_indices:
                raise ValueError(
                    f"Selected page numbers {page_numbers} out of bounds (document has {total_doc_pages} pages)"
                )
        else:
            target_indices = list(range(1, total_doc_pages + 1))

        for idx in target_indices:
            page_filename = f"{stem}_page-{idx:04d}.png"
            page_out_path = output_dir / page_filename

            # Extract only this single page on the fly
            with tempfile.TemporaryDirectory(prefix=f"ocr_p{idx}_") as td:
                temp_page = Path(td) / "page.png"
                extract_pdf_page(src_path, idx, temp_page, dpi=dpi)

                manifest_entry = prepare_page(
                    image_path=temp_page,
                    output_path=page_out_path,
                    page_index=idx,
                    force_ops=force_ops,
                    source_label=str(src_path),
                    run_ocr=run_ocr,
                    ocr_lang=ocr_lang,
                )
                pages.append(manifest_entry)
                _sync_manifest()

                if callback:
                    callback(manifest_entry)

                yield manifest_entry

    else:
        page_filename = f"{stem}_page-0001.png"
        page_out_path = output_dir / page_filename
        manifest_entry = prepare_page(
            image_path=src_path,
            output_path=page_out_path,
            page_index=1,
            force_ops=force_ops,
            source_label=str(src_path),
            run_ocr=run_ocr,
            ocr_lang=ocr_lang,
        )
        pages.append(manifest_entry)
        _sync_manifest()

        if callback:
            callback(manifest_entry)

        yield manifest_entry


def prepare_document(
    src: str | Path,
    out_dir: str | Path,
    dpi: int = 300,
    force_ops: list[str] | None = None,
    verbose: bool = True,
    page_numbers: list[int] | None = None,
    run_ocr: bool = False,
    ocr_lang: str = "kan",
    callback: callable | None = None,
) -> DocManifest:
    """
    Process a PDF or single image into OCR-ready images and a JSON manifest.

    Pages are processed one-by-one in a streaming fashion. If run_ocr is enabled,
    OCR is dispatched on each page immediately upon preprocessing completion.
    """
    src_path = Path(src)
    output_dir = Path(out_dir)
    stem = src_path.stem
    manifest_filename = f"{stem}_manifest.json"

    pages: list[PageManifest] = []
    total_doc_pages = get_pdf_page_count(src_path) if src_path.suffix.lower() in PDF_EXTENSIONS else 1
    total_target = len(page_numbers) if page_numbers else total_doc_pages
    w = max(len(str(total_doc_pages)), 3)

    stream = stream_prepare_document(
        src=src,
        out_dir=out_dir,
        dpi=dpi,
        force_ops=force_ops,
        page_numbers=page_numbers,
        run_ocr=run_ocr,
        ocr_lang=ocr_lang,
        callback=callback,
    )

    for count_idx, entry in enumerate(stream, start=1):
        pages.append(entry)
        if verbose:
            ops_str = (
                f"ops={','.join(entry.operations)}"
                if entry.operations
                else "ops=none"
            )
            ocr_info = f" -> OCR ({len(entry.ocr_text)} chars)" if run_ocr else ""
            print(
                f"[{count_idx:{w}d}/{total_target:{w}d}] (p.{entry.page_index}) "
                f"{entry.decision:<10}  "
                f"{ops_str:<40} -> "
                f"{entry.output}{ocr_info}"
            )

    doc_manifest = DocManifest(
        source=str(src_path),
        pages=pages,
    )

    if verbose:
        out_display = str(output_dir).rstrip("/") + "/"
        print(
            f"wrote {len(pages)} page(s) + {manifest_filename} to {out_display}"
        )
        print(f"  preprocessed: {doc_manifest.n_preprocessed}")
        print(f"  kept original: {doc_manifest.n_kept}")

    return doc_manifest


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
        "--page",
        type=int,
        default=None,
        help="Process only a single 1-indexed page (e.g. --page 1)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Comma-separated pages or range to process (e.g. --pages 1-5 or --pages 1,3,5)",
    )
    parser.add_argument(
        "--force-ops",
        type=str,
        default=None,
        help="Comma-separated operations to force (empty string forces KEEP)",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Immediately run downstream OCR (Tesseract) on each page as soon as it is preprocessed",
    )
    parser.add_argument(
        "--ocr-lang",
        type=str,
        default="kan",
        help="Language code for OCR (default: 'kan')",
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

    page_numbers = None
    if args.page is not None:
        page_numbers = [args.page]
    elif args.pages is not None:
        page_numbers = parse_page_selection(args.pages, 1000000)

    prepare_document(
        src=args.input,
        out_dir=args.output,
        dpi=args.dpi,
        force_ops=force_ops,
        verbose=not args.quiet,
        page_numbers=page_numbers,
        run_ocr=args.ocr,
        ocr_lang=args.ocr_lang,
    )


if __name__ == "__main__":
    main()
