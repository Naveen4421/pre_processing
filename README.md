# Kannada OCR Preprocessing Pipeline & Dev Lab

A quality-aware, non-destructive document scan preprocessing pipeline and developer inspection tool designed for Indic script OCR, specifically tuned for Kannada book and document pages.

---

## Overview

Traditional OCR preprocessing pipelines often blindly apply destructive binarization, aggressive thresholding, or high-pass filtering across all pages. For Indic scripts such as Kannada, this indiscriminate filtering causes **irreversible damage**:
- Delicate loops and rounded ligatures get broken or filled in.
- Subtle subscript consonants (*ottu / vatlu*) and vowel signs (*matras*) disappear.
- Clean, high-contrast scans end up degraded compared to their raw state.

This project introduces a **Quality Gate Architecture**:
1. **Analyze First**: Every input page is evaluated across multiple image quality metrics (skew angle, blur/sharpness, contrast, illumination uniformity, and high-frequency noise).
2. **Selective Intervention**: Clean pages are preserved **untouched** to maintain character stroke geometry. Degraded pages only receive **the specific corrective operations** they require.
3. **OCR Handoff Ready**: Exports normalized, zero-padded grayscale page images along with an indexed `manifest.json` describing metrics and operations applied.
4. **Developer Inspection Lab**: Includes a local Flask web app for visual verification, multi-mode image comparison (side-by-side, overlay, blink), and interactive per-page A/B testing.

---

## Pipeline Architecture

```
                       Input Document (PDF / Image)
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │                 Quality Gate Assessment                 │
        │  • Blur (Laplacian Variance)                            │
        │  • Contrast (Std Dev)                                   │
        │  • Skew (Hough Transform Angle)                         │
        │  • Illumination (Background Std)                        │
        │  • Noise (High-Frequency Absdiff)                       │
        └────────────────────────────┬────────────────────────────┘
                                     │
                         Needs Preprocessing?
                        /                    \
                [No]   /                      \   [Yes]
                      ▼                        ▼
             Preserve Original        Selective Transformations
             (Byte Passthrough)       (Fixed Safe Order)
                                      1. crop
                                      2. deskew
                                      3. illumination
                                      4. denoise
                                      5. contrast
                                      6. sharpen
                      │                        │
                      └──────────────┬─────────┘
                                     ▼
                      Normalized Grayscale Pages
                                     +
                           manifest.json Output
```

### Why Operations Run in a Fixed Safe Order

`OP_ORDER = ["crop", "deskew", "illumination", "denoise", "contrast", "sharpen"]`

1. **`crop` first**: Non-document backgrounds (table surfaces, cloth, shadows, camera margins) must be excised before geometry correction and illumination normalization, ensuring subsequent filters only analyze the actual page content.
2. **`deskew` second**: Geometric orientation must be corrected before spatial filtering and intensity mapping so that coordinate spaces and orientation are normalized.
3. **`illumination` before `contrast`**: Large-scale background gradients and book-fold shadows must be flattened before local equalization; otherwise, CLAHE amplifies background unevenness into dark or blown-out patches.
4. **`denoise` before `contrast`**: Paper and sensor grain must be smoothed (via an edge-preserving bilateral filter) before contrast stretching to prevent noise amplification.
5. **`contrast`**: CLAHE improves local character contrast while guarding thin strokes.
6. **`sharpen` last**: Unsharp masking must run at the very end; executing it earlier would accentuate high-frequency noise throughout prior stages.

---

## Quality Gate Metrics & Operations

The quality gate evaluates images using OpenCV and decides which corrective actions are needed:

| Metric | Measurement Technique | Threshold Condition | Triggered Operation | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Boundary Crop** | Canny edges + morphological dilation + 4-point polygon approximation (`approxPolyDP`) | Detected document quad with $0.20 \le \text{area ratio} < 0.95$ | `crop` | Dynamically trims foreign background (table, cloth, shadows) and applies 4-point perspective rectification. Flatbed scans remain 100% untouched. |
| **Skew** | Canny edges + Probabilistic Hough Lines (`HoughLinesP`) filtering $\pm 15^\circ$ | `abs(skew_angle) > 1.5°` | `deskew` | Rotates image around center with `INTER_CUBIC` and clean white border padding keeping canvas size. |
| **Illumination** | Downsampled Gaussian blur standard deviation | `illumination_score > 18.0` | `illumination` | Gaussian-division background normalization (`sigmaX=35`) to remove lighting gradients. |
| **Noise** | High-frequency residual mean (`cv2.absdiff` with $3\times3$ blur) | `noise_score > 8.0` | `denoise` | Bilateral filter (`d=7, sigma=35`) preserving character edges without destroying *matras*. |
| **Blur** | Variance of the Laplacian (`cv2.Laplacian`) | `blur_score < 600.0` | `sharpen` | Unsharp mask (`sigma=1.0, amount=0.8`) to restore soft text without halo artifacts. |
| **Contrast** | Grayscale intensity standard deviation | `contrast_score < 35.0` | `contrast` | CLAHE (`clipLimit=2.5, tileGridSize=(8,8)`) enhancing faint ink while avoiding stroke blowout. |

---

## Project Structure

```
.
├── input/                          # Sample input document scans (e.g., page001.png, page002.png)
├── output/                         # Processed and batch outputs
├── preprocessing/                  # Core processing package
│   ├── __init__.py
│   ├── quality_gate.py             # Page quality assessment & decision engine
│   ├── kannada_preprocess.py       # Selective transformation orchestrator (5 operations)
│   ├── prepare_for_ocr.py          # PDF/image OCR handoff tool (manifest generator)
│   ├── auto_preprocess.py          # Batch & single-image CLI pipeline
│   ├── contrast_test.py            # Standalone CLAHE contrast test
│   └── illumination_test.py        # Standalone Gaussian illumination test
├── webapp/                         # Local developer inspection & A/B testing web app
│   ├── app.py                      # Flask server (upload, gallery, viewer, rerun endpoints)
│   ├── static/
│   │   └── style.css               # Minimal dark theme (<150 lines, CSS variables)
│   ├── templates/
│   │   ├── index.html              # Upload & thumbnail gallery template
│   │   └── viewer.html             # Multi-mode side-by-side inspector & A/B rerun tool
│   ├── uploads/                    # User session uploads (gitignored)
│   └── outputs/                    # Session outputs, thumbnails & manifests (gitignored)
├── requirements.txt                # Python dependencies (numpy, opencv, flask)
├── .python-version                 # Supported Python version (3.10+)
└── README.md                       # Documentation
```

---

## Installation & Setup

### Prerequisites
- **Python**: 3.10 or higher
- **System Utilities** (for PDF rasterization):
  - Ubuntu/Debian: `sudo apt install poppler-utils`
  - macOS (Homebrew): `brew install poppler`
  - Arch Linux: `sudo pacman -S poppler`

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Naveen4421/pre_processing.git
   cd pre_processing
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide

### 1. Developer Inspection Web App (Interactive UI)

Launch the local Flask application for visual inspection, multi-mode comparisons, and per-page A/B testing:

```bash
python -m webapp.app
```
Open **`http://127.0.0.1:5000`** in your browser.

#### Key Features:
- **Upload Flow**: Supports PDF books or individual images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.pnm`) with optional operation overrides.
- **Instant PDF Streaming**: Multi-hundred page PDF books ingest in <1 second. Page 1 is ready immediately, and subsequent pages are rendered and preprocessed on-demand as you browse.
- **Gallery Grid & Page Jumper**: Displays page cards with live `KEEP`, `PREPROCESS`, and `PENDING` statuses; jump directly to any page or click **Process Next Page** to walk through pages sequentially.
- **Multi-Mode Inspector**:
  - `[1] Side-by-Side`: Default side-by-side layout of raw vs. processed pages.
  - `[2] Overlay`: Absolutely positioned processed image over the raw image at 50% opacity.
  - `[3] Blink`: Alternates between raw and processed images every 600ms to immediately highlight stroke changes.
  - `[4] Processed Only`: Focused single-image inspection.
- **Keyboard Navigation**: Use `←` / `→` arrows to navigate pages; use keys `1`–`4` to switch view modes.
- **Interactive A/B Testing**: Directly modify operations in the web UI and hit **Re-run Page** to re-execute transformations on that raw page, overwrite the processed image, and update the manifest.
- **Manifest Export**: Direct link to download `manifest.json` for downstream OCR ingestion.

---

### 2. OCR Handoff Pipeline (`prepare_for_ocr.py`)

Prepares a PDF document or single image for downstream OCR engines (such as Tesseract, PaddleOCR, or custom models):

```bash
# Process a PDF document at 300 DPI (page-by-page streaming)
python -m preprocessing.prepare_for_ocr book.pdf output/book/ --dpi 300

# Process only a single page from a large multi-page PDF
python -m preprocessing.prepare_for_ocr book.pdf output/book/ --page 1

# Process a specific page range or selection (e.g. pages 1 to 5, or 1,3,5)
python -m preprocessing.prepare_for_ocr book.pdf output/book/ --pages 1-5

# Process a single image
python -m preprocessing.prepare_for_ocr input/page001.png output/page001/

# Force specific operations across all pages (bypassing the quality gate)
python -m preprocessing.prepare_for_ocr book.pdf output/book/ --force-ops deskew,contrast

# Suppress console progress
python -m preprocessing.prepare_for_ocr book.pdf output/book/ --quiet
```

#### Output Structure:
```
output/book/
├── book_page-0001.png
├── book_page-0002.png
├── book_page-0003.png
└── book_manifest.json
```

#### Sample `book_manifest.json`:
```json
{
  "source": "book.pdf",
  "pages": [
    {
      "page_index": 1,
      "source": "book.pdf",
      "output": "book_page-0001.png",
      "width": 1465,
      "height": 1800,
      "decision": "PREPROCESS",
      "operations": [
        "deskew"
      ],
      "metrics": {
        "skew_deg": -3.95,
        "blur": 4030.3,
        "contrast": 76.85,
        "illumination": 11.9,
        "noise": 5.6
      }
    }
  ]
}
```

---

### 3. Automated Batch & Single-File CLI (`auto_preprocess.py`)

Run quality-aware preprocessing on individual files or entire directories:

```bash
# Single image
python -m preprocessing.auto_preprocess input/page001.png output/page001_auto.png

# Process entire directory
python -m preprocessing.auto_preprocess input/ output/

# Recursive batch with quiet flag
python -m preprocessing.auto_preprocess input/ output/ --recursive --quiet
```

**Sample Output (`page001.png` - Skewed):**
```text
========== PAGE QUALITY ==========
Image        : input/page001.png
Resolution   : 1465 x 1800
Blur         : 4030.3
Contrast     : 76.85
Skew         : -3.95°
Illumination : 11.9
Noise        : 5.6

Needs preprocessing: True
Operations:
  - deskew
==================================

========== AUTO PREPROCESS ==========
Input      : input/page001.png
Decision   : PREPROCESS
Operations : ['deskew']
Output     : output/page001_auto.png
=====================================
```

**Sample Output (`page002.png` - Clean Page):**
```text
========== PAGE QUALITY ==========
Image        : input/page002.png
Resolution   : 990 x 863
Blur         : 1595.23
Contrast     : 37.6
Skew         : -1.01°
Illumination : 9.75
Noise        : 3.0

Needs preprocessing: False
Operations: NONE
==================================

========== AUTO PREPROCESS ==========
Input      : input/page002.png
Decision   : KEEP ORIGINAL
Operations : NONE
Output     : output/page002_auto.png
=====================================
```

---

### 4. Standalone Quality Gate Assessment (`quality_gate.py`)

Inspect image metrics without altering files:

```bash
python -m preprocessing.quality_gate input/page001.png
```

---

## Python API Integration

Integrate the quality gate and preprocessing pipeline directly into Python applications:

```python
import cv2
from pathlib import Path
from preprocessing.prepare_for_ocr import prepare_document, prepare_page
from preprocessing.quality_gate import assess_page
from preprocessing.kannada_preprocess import preprocess_page

# 1. High-level document preparation (generates normalized pages + manifest)
manifest = prepare_document(
    src="book.pdf",
    out_dir="output/book",
    dpi=300,
)
print(f"Prepared {len(manifest.pages)} pages: {manifest.n_preprocessed} preprocessed, {manifest.n_kept} kept original.")

# 2. In-memory page assessment & selective preprocessing
image = cv2.imread("input/page001.png", cv2.IMREAD_GRAYSCALE)
quality = assess_page(image)

if quality.needs_preprocessing:
    cleaned = preprocess_page(
        image,
        skew_angle=quality.skew_angle,
        operations=quality.operations,
    )
else:
    cleaned = image  # preserve byte-for-byte
```

---

## Design Principles

1. **Non-Destructive Grayscale Preservation**: Avoid premature binarization. Standard Otsu or Sauvola binarization can break delicate loops and subscript ligatures in Kannada. Grayscale images preserve the sub-pixel stroke anti-aliasing needed by modern neural OCR models.
2. **Fixed Safe Pipeline Order**: Operations are executed strictly in boundary crop $\rightarrow$ geometry (deskew) $\rightarrow$ illumination $\rightarrow$ denoise $\rightarrow$ contrast $\rightarrow$ sharpen order to prevent artifact cascades.
3. **Reproducibility & Traceability**: The manifest records exact numerical metrics and transformation decisions for every page, enabling automated quality auditing and retraining data generation.

---

## License

This project is licensed under the [MIT License](LICENSE).
