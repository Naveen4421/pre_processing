"""
Kannada OCR Preprocessing Development & Inspection Web Application.

A lightweight Flask tool for developers to upload PDF and image scans, inspect
automated quality gate assessments and non-destructive transformations page-by-page,
and perform interactive A/B testing with custom operation overrides.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import cv2
from flask import Flask, abort, redirect, render_template, request, send_file, url_for
import numpy as np

from preprocessing.kannada_preprocess import preprocess_page
from preprocessing.prepare_for_ocr import prepare_document
from preprocessing.quality_gate import assess_page

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".pnm",
}


def _validate_uuid(session_id: str) -> str:
    """Validate and sanitize UUID string to prevent path traversal."""
    return str(uuid.UUID(str(session_id).strip()))


def _get_manifest_path(session_dir: Path) -> Path | None:
    """Find the manifest JSON file in a session directory."""
    direct = session_dir / "manifest.json"
    if direct.is_file():
        return direct
    matches = list(session_dir.glob("*_manifest.json"))
    if matches:
        return matches[0]
    return None


def _load_manifest(session_id: str) -> dict | None:
    """Load session manifest dictionary from disk."""
    s_id = _validate_uuid(session_id)
    session_dir = OUTPUTS_DIR / s_id
    manifest_file = _get_manifest_path(session_dir)
    if not manifest_file or not manifest_file.is_file():
        return None
    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(session_id: str, data: dict) -> None:
    """Persist updated manifest to manifest.json and any original manifest copy."""
    s_id = _validate_uuid(session_id)
    session_dir = OUTPUTS_DIR / s_id
    target = session_dir / "manifest.json"
    json_bytes = json.dumps(data, indent=2, ensure_ascii=False)
    target.write_text(json_bytes, encoding="utf-8")

    # Also sync original stem manifest if it exists
    for existing in session_dir.glob("*_manifest.json"):
        if existing.name != "manifest.json":
            existing.write_text(json_bytes, encoding="utf-8")


def _make_thumbnail(src_path: Path, thumb_path: Path, target_width: int = 240) -> None:
    """Downscale processed page image for fast gallery rendering using OpenCV."""
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src_path))
    if img is None:
        return
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return
    target_height = max(1, int(h * (target_width / w)))
    thumb = cv2.resize(
        img, (target_width, target_height), interpolation=cv2.INTER_AREA
    )
    cv2.imwrite(str(thumb_path), thumb)


@app.route("/")
def index():
    """Render landing page with upload form."""
    return render_template("index.html", session_id=None, manifest=None)


@app.route("/upload", methods=["POST"])
def upload():
    """Handle document upload and initiate preprocessing pipeline."""
    if "file" not in request.files:
        abort(400, description="No file uploaded")

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        abort(400, description="Empty filename")

    filename = Path(uploaded_file.filename).name
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        abort(
            400,
            description=f"Unsupported extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    session_id = str(uuid.uuid4())
    upload_dir = UPLOADS_DIR / session_id
    out_dir = OUTPUTS_DIR / session_id
    raw_dir = out_dir / "raw"

    upload_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    upload_path = upload_dir / filename
    uploaded_file.save(str(upload_path))

    # Parse optional force_ops
    force_ops_val = request.form.get("force_ops", "").strip()
    force_ops = (
        [x.strip() for x in force_ops_val.split(",") if x.strip()]
        if force_ops_val
        else None
    )

    # Save raw page images into <outputs>/<uuid>/raw/ so they remain available
    # for side-by-side comparison in the viewer.
    # For PDFs, invoke pdftoppm directly into raw/ before calling prepare_document.
    # This uses two subprocess runs but avoids modifying prepare_document.
    if ext == ".pdf":
        if shutil.which("pdftoppm") is None:
            abort(
                500,
                description="pdftoppm not found. Please install poppler-utils.",
            )
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                "300",
                str(upload_path),
                str(raw_dir / "page"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        raw_files = sorted(
            raw_dir.glob("*.png"),
            key=lambda p: int(p.stem.split("-")[-1])
            if p.stem.split("-")[-1].isdigit()
            else p.stem,
        )
        for idx, rf in enumerate(raw_files, start=1):
            target = raw_dir / f"page_{idx}.png"
            if rf != target:
                rf.rename(target)
    else:
        # For single image input, save directly as raw/page_1.png
        shutil.copy2(upload_path, raw_dir / "page_1.png")

    # Run the core preprocessing pipeline
    prepare_document(
        src=upload_path,
        out_dir=out_dir,
        dpi=300,
        force_ops=force_ops,
        verbose=False,
    )

    # Ensure standardized manifest.json exists in session output folder
    orig_manifest = _get_manifest_path(out_dir)
    std_manifest = out_dir / "manifest.json"
    if orig_manifest and orig_manifest.is_file() and orig_manifest != std_manifest:
        shutil.copy2(orig_manifest, std_manifest)

    return redirect(url_for("gallery", session_id=session_id))


@app.route("/session/<uuid:session_id>")
def gallery(session_id):
    """Render thumbnail gallery for an uploaded document session."""
    s_id = str(session_id)
    manifest = _load_manifest(s_id)
    if not manifest:
        abort(404, description="Session manifest not found")

    n_preprocessed = sum(
        1 for p in manifest.get("pages", []) if p.get("decision") == "PREPROCESS"
    )
    n_kept = sum(
        1 for p in manifest.get("pages", []) if p.get("decision") == "KEEP"
    )

    return render_template(
        "index.html",
        session_id=s_id,
        manifest=manifest,
        n_preprocessed=n_preprocessed,
        n_kept=n_kept,
    )


@app.route("/session/<uuid:session_id>/page/<int:page_n>")
def page_viewer(session_id, page_n):
    """Render per-page side-by-side inspector with comparison modes and A/B rerun form."""
    s_id = str(session_id)
    manifest = _load_manifest(s_id)
    if not manifest:
        abort(404, description="Session not found")

    pages = manifest.get("pages", [])
    total_pages = len(pages)
    if page_n < 1 or page_n > total_pages:
        abort(404, description="Page index out of bounds")

    page = pages[page_n - 1]
    return render_template(
        "viewer.html",
        session_id=s_id,
        manifest=manifest,
        page=page,
        page_n=page_n,
        total_pages=total_pages,
        cache_bust=int(time.time()),
    )


@app.route("/session/<uuid:session_id>/page/<int:page_n>/rerun", methods=["POST"])
def rerun_page(session_id, page_n):
    """Re-run preprocessing on a single page with custom operation overrides."""
    s_id = str(session_id)
    manifest = _load_manifest(s_id)
    if not manifest:
        abort(404, description="Session not found")

    pages = manifest.get("pages", [])
    if page_n < 1 or page_n > len(pages):
        abort(404, description="Page index out of bounds")

    page_entry = pages[page_n - 1]
    ops_str = request.form.get("ops", "")
    ops = [x.strip() for x in ops_str.split(",") if x.strip()]

    # Load original raw page image
    raw_path = OUTPUTS_DIR / s_id / "raw" / f"page_{page_n}.png"
    if not raw_path.is_file():
        abort(404, description="Raw page image not found")

    raw_img = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        abort(500, description="Could not load raw page image")

    # Assess quality for angle parameter
    q = assess_page(raw_img)

    # Conditionally execute requested operations
    if ops:
        processed = preprocess_page(
            raw_img, skew_angle=q.skew_angle, operations=ops
        )
        decision = "PREPROCESS"
    else:
        processed = raw_img
        decision = "KEEP"

    # Overwrite the processed image on disk
    out_filename = Path(page_entry["output"]).name
    out_path = OUTPUTS_DIR / s_id / out_filename
    if not cv2.imwrite(str(out_path), processed):
        abort(500, description="Failed to overwrite processed page image")

    # Invalidate thumbnail cache so gallery updates
    thumb_path = OUTPUTS_DIR / s_id / "thumbs" / out_filename
    if thumb_path.is_file():
        thumb_path.unlink()

    # Update manifest entry and persist
    page_entry["decision"] = decision
    page_entry["operations"] = ops
    page_entry["width"] = int(processed.shape[1])
    page_entry["height"] = int(processed.shape[0])
    _save_manifest(s_id, manifest)

    return redirect(url_for("page_viewer", session_id=s_id, page_n=page_n))


@app.route("/session/<uuid:session_id>/img/raw/<int:page_n>")
def get_raw_image(session_id, page_n):
    """Safely serve the uncompressed raw page image."""
    s_id = str(session_id)
    manifest = _load_manifest(s_id)
    if not manifest or page_n < 1 or page_n > len(manifest.get("pages", [])):
        abort(404)
    raw_path = OUTPUTS_DIR / s_id / "raw" / f"page_{page_n}.png"
    if not raw_path.is_file():
        abort(404)
    return send_file(raw_path, mimetype="image/png")


@app.route("/session/<uuid:session_id>/img/processed/<int:page_n>")
def get_processed_image(session_id, page_n):
    """Safely serve the processed page image referenced by the manifest."""
    s_id = str(session_id)
    manifest = _load_manifest(s_id)
    if not manifest or page_n < 1 or page_n > len(manifest.get("pages", [])):
        abort(404)
    page_entry = manifest["pages"][page_n - 1]
    out_filename = Path(page_entry["output"]).name
    proc_path = OUTPUTS_DIR / s_id / out_filename
    if not proc_path.is_file():
        abort(404)
    return send_file(proc_path, mimetype="image/png")


@app.route("/session/<uuid:session_id>/img/thumb/<int:page_n>")
def get_thumbnail(session_id, page_n):
    """Serve or generate cached 240px thumbnail for gallery view."""
    s_id = str(session_id)
    manifest = _load_manifest(s_id)
    if not manifest or page_n < 1 or page_n > len(manifest.get("pages", [])):
        abort(404)
    page_entry = manifest["pages"][page_n - 1]
    out_filename = Path(page_entry["output"]).name
    proc_path = OUTPUTS_DIR / s_id / out_filename
    if not proc_path.is_file():
        abort(404)

    thumbs_dir = OUTPUTS_DIR / s_id / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumbs_dir / out_filename
    if not thumb_path.is_file():
        _make_thumbnail(proc_path, thumb_path, target_width=240)

    return send_file(thumb_path, mimetype="image/png")


@app.route("/session/<uuid:session_id>/manifest.json")
def download_manifest(session_id):
    """Download raw session manifest JSON."""
    s_id = str(session_id)
    session_dir = OUTPUTS_DIR / s_id
    manifest_path = _get_manifest_path(session_dir)
    if not manifest_path or not manifest_path.is_file():
        abort(404)
    return send_file(
        manifest_path, mimetype="application/json", as_attachment=True
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
