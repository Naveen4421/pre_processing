"""
Kannada Document Preprocessing Development & Inspection Web Application.

A lightweight Flask tool for developers to upload PDF and image scans, inspect
automated quality gate assessments and non-destructive transformations page-by-page,
and perform interactive A/B testing with custom operation overrides. Produces
OCR-ready images and a manifest for a downstream OCR stage; OCR itself is out
of scope here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid

import cv2
from flask import Flask, abort, redirect, render_template, request, send_file, url_for
import numpy as np

from preprocessing.kannada_preprocess import preprocess_page
from preprocessing.prepare_for_ocr import (
    extract_pdf_page,
    get_pdf_page_count,
    prepare_document,
    prepare_page,
)
from preprocessing.quality_gate import assess_page

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

_ACTIVE_STREAMS: dict[str, bool] = {}
_STREAM_LOCK = threading.Lock()

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


def _process_single_page(session_id: str, page_n: int, run_ocr: bool = False) -> dict:
    """Extract and preprocess a single page for a session (OCR disabled)."""
    s_id = _validate_uuid(session_id)
    manifest = _load_manifest(s_id)
    if not manifest:
        abort(404, description="Session manifest not found")

    pages = manifest.get("pages", [])
    if page_n < 1 or page_n > len(pages):
        abort(404, description="Page index out of bounds")

    upload_dir = UPLOADS_DIR / s_id
    out_dir = OUTPUTS_DIR / s_id
    raw_dir = out_dir / "raw"
    thumbs_dir = out_dir / "thumbs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    page_entry = pages[page_n - 1]
    raw_path = raw_dir / f"page_{page_n}.png"

    # Find the uploaded source file
    upload_files = list(upload_dir.iterdir())
    if not upload_files:
        abort(500, description="Original upload file missing")
    upload_path = upload_files[0]
    ext = upload_path.suffix.lower()

    if not raw_path.is_file():
        if ext == ".pdf":
            extract_pdf_page(upload_path, page_n, raw_path, dpi=300)
        else:
            shutil.copy2(upload_path, raw_path)

    out_filename = page_entry.get("output") or f"{upload_path.stem}_page-{page_n:04d}.png"
    out_path = out_dir / out_filename

    force_ops = manifest.get("force_ops")
    pm = prepare_page(
        image_path=raw_path,
        output_path=out_path,
        page_index=page_n,
        force_ops=force_ops,
        source_label=str(upload_path),
        run_ocr=run_ocr,
    )

    thumb_path = thumbs_dir / out_filename
    _make_thumbnail(out_path, thumb_path, target_width=240)

    updated_entry = {
        "page_index": pm.page_index,
        "source": pm.source,
        "output": pm.output,
        "width": pm.width,
        "height": pm.height,
        "decision": pm.decision,
        "operations": pm.operations,
        "metrics": pm.metrics,
        "ocr_text": pm.ocr_text,
    }
    manifest["pages"][page_n - 1] = updated_entry
    _save_manifest(s_id, manifest)
    return updated_entry


def _stream_worker(session_id: str):
    """Background worker that continuously preprocesses pending pages."""
    with _STREAM_LOCK:
        if _ACTIVE_STREAMS.get(session_id):
            return
        _ACTIVE_STREAMS[session_id] = True

    try:
        while True:
            manifest = _load_manifest(session_id)
            if not manifest:
                break
            pending = [
                p["page_index"]
                for p in manifest.get("pages", [])
                if p.get("decision") == "PENDING"
            ]
            if not pending:
                break
            next_page = pending[0]
            try:
                _process_single_page(session_id, next_page, run_ocr=False)
            except Exception as err:
                print(f"Background streaming error on page {next_page}: {err}")
                break
    finally:
        with _STREAM_LOCK:
            _ACTIVE_STREAMS[session_id] = False


def _start_background_stream(session_id: str):
    """Launch background daemon thread for sequential page-by-page preprocessing."""
    t = threading.Thread(target=_stream_worker, args=(session_id,), daemon=True)
    t.start()


@app.route("/")
def index():
    """Render landing page with upload form."""
    return render_template("index.html", session_id=None, manifest=None)


@app.route("/upload", methods=["POST"])
def upload():
    """Handle document upload with fast 1-page-at-a-time streaming for PDFs."""
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
    thumbs_dir = out_dir / "thumbs"

    upload_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    upload_path = upload_dir / filename
    uploaded_file.save(str(upload_path))

    # Parse optional force_ops
    force_ops_val = request.form.get("force_ops", "").strip()
    force_ops = (
        [x.strip() for x in force_ops_val.split(",") if x.strip()]
        if force_ops_val
        else None
    )

    if ext == ".pdf":
        if shutil.which("pdftoppm") is None:
            abort(
                500,
                description="pdftoppm not found. Please install poppler-utils.",
            )
        total_pages = get_pdf_page_count(upload_path)
        stem = upload_path.stem

        # Extract & prepare Page 1 immediately for instant preview (< 0.5s)
        p1_raw = raw_dir / "page_1.png"
        extract_pdf_page(upload_path, 1, p1_raw, dpi=300)
        p1_out_filename = f"{stem}_page-0001.png"
        p1_out = out_dir / p1_out_filename
        p1_entry = prepare_page(
            image_path=p1_raw,
            output_path=p1_out,
            page_index=1,
            force_ops=force_ops,
            source_label=str(upload_path),
            run_ocr=False,
        )
        _make_thumbnail(p1_out, thumbs_dir / p1_out_filename, target_width=240)

        pages_list = [
            {
                "page_index": p1_entry.page_index,
                "source": p1_entry.source,
                "output": p1_entry.output,
                "width": p1_entry.width,
                "height": p1_entry.height,
                "decision": p1_entry.decision,
                "operations": p1_entry.operations,
                "metrics": p1_entry.metrics,
                "ocr_text": p1_entry.ocr_text,
            }
        ]

        # Remaining pages are initialized as PENDING and streamed in background
        for idx in range(2, total_pages + 1):
            pages_list.append({
                "page_index": idx,
                "source": str(upload_path),
                "output": f"{stem}_page-{idx:04d}.png",
                "width": 0,
                "height": 0,
                "decision": "PENDING",
                "operations": [],
                "metrics": {},
                "ocr_text": "",
            })

        manifest_data = {
            "source": str(upload_path),
            "total_pages": total_pages,
            "force_ops": force_ops,
            "pages": pages_list,
        }
        _save_manifest(session_id, manifest_data)

        # Start continuous background streaming: Page 1 is ready, Pages 2..N are processed on-demand
        _start_background_stream(session_id)

    else:
        # Single image input
        shutil.copy2(upload_path, raw_dir / "page_1.png")
        prepare_document(
            src=upload_path,
            out_dir=out_dir,
            dpi=300,
            force_ops=force_ops,
            verbose=False,
            run_ocr=False,
        )
        orig_manifest = _get_manifest_path(out_dir)
        std_manifest = out_dir / "manifest.json"
        if orig_manifest and orig_manifest.is_file() and orig_manifest != std_manifest:
            shutil.copy2(orig_manifest, std_manifest)

    return redirect(url_for("page_viewer", session_id=session_id, page_n=1))


@app.route("/session/<uuid:session_id>/status")
def session_status(session_id):
    """Return JSON status of the background streaming pipeline."""
    s_id = str(session_id)
    manifest = _load_manifest(s_id)
    if not manifest:
        abort(404)
    pages = manifest.get("pages", [])
    n_pre = sum(1 for p in pages if p.get("decision") == "PREPROCESS")
    n_kept = sum(1 for p in pages if p.get("decision") == "KEEP")
    n_pending = sum(1 for p in pages if p.get("decision") == "PENDING")
    latest_done = 0
    for p in pages:
        if p.get("decision") != "PENDING":
            latest_done = max(latest_done, p["page_index"])

    with _STREAM_LOCK:
        is_active = bool(_ACTIVE_STREAMS.get(s_id))

    return {
        "session_id": s_id,
        "total_pages": len(pages),
        "preprocessed": n_pre,
        "kept": n_kept,
        "pending": n_pending,
        "latest_done": latest_done,
        "is_streaming": is_active,
    }


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
    n_pending = sum(
        1 for p in manifest.get("pages", []) if p.get("decision") == "PENDING"
    )

    return render_template(
        "index.html",
        session_id=s_id,
        manifest=manifest,
        n_preprocessed=n_preprocessed,
        n_kept=n_kept,
        n_pending=n_pending,
    )


@app.route("/session/<uuid:session_id>/process_next")
def process_next(session_id):
    """Process the next pending page and open it in the viewer."""
    s_id = str(session_id)
    manifest = _load_manifest(s_id)
    if not manifest:
        abort(404)
    for p in manifest.get("pages", []):
        if p.get("decision") == "PENDING":
            _process_single_page(s_id, p["page_index"])
            return redirect(url_for("page_viewer", session_id=s_id, page_n=p["page_index"]))
    return redirect(url_for("gallery", session_id=s_id))


@app.route("/session/<uuid:session_id>/page/<int:page_n>")
def page_viewer(session_id, page_n):
    """Render per-page side-by-side inspector with on-demand single page processing."""
    s_id = str(session_id)
    manifest = _load_manifest(s_id)
    if not manifest:
        abort(404, description="Session not found")

    pages = manifest.get("pages", [])
    total_pages = len(pages)
    if page_n < 1 or page_n > total_pages:
        abort(404, description="Page index out of bounds")

    page_entry = pages[page_n - 1]
    # If the page has not been processed yet, process it on-demand in <0.5s
    if page_entry.get("decision") == "PENDING" or not (OUTPUTS_DIR / s_id / "raw" / f"page_{page_n}.png").is_file():
        page_entry = _process_single_page(s_id, page_n)
        manifest = _load_manifest(s_id)

    return render_template(
        "viewer.html",
        session_id=s_id,
        manifest=manifest,
        page=page_entry,
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

    # Load original raw page image (extract if pending)
    raw_path = OUTPUTS_DIR / s_id / "raw" / f"page_{page_n}.png"
    if not raw_path.is_file():
        _process_single_page(s_id, page_n)
        manifest = _load_manifest(s_id)
        page_entry = manifest["pages"][page_n - 1]

    raw_img = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        abort(500, description="Could not load raw page image")

    # Assess quality for angle parameters
    q = assess_page(raw_img)

    # Conditionally execute requested operations
    if ops:
        processed = preprocess_page(
            raw_img,
            skew_angle=q.skew_angle,
            orientation_angle=q.orientation_angle,
            operations=ops,
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
        _process_single_page(s_id, page_n)
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
    if not proc_path.is_file() or page_entry.get("decision") == "PENDING":
        page_entry = _process_single_page(s_id, page_n)
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
    if page_entry.get("decision") == "PENDING":
        abort(404)  # UI renders pending card placeholder instead of loading image
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
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "1") == "1",
    )
