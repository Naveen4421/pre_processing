# Kannada OCR Preprocessing Pipeline

A quality-aware, non-destructive document image preprocessing pipeline designed for OCR, with tuning specifically tailored for Kannada book and document pages.

---

## Overview

Traditional OCR preprocessing pipelines often blindly apply heavy binarization, aggressive thresholding, and morphological filtering across all pages. For Indic scripts such as Kannada, this indiscriminate filtering frequently causes **irreversible damage**:
- Delicate loops and ligatures get broken or filled in.
- Subtle subscript characters (*ottu / vatlu*) and vowel signs (*matras*) disappear.
- Clean, high-contrast scans end up degraded compared to their raw state.

This project introduces a **Quality Gate Architecture**:
1. **Analyze First**: Every input page is evaluated across multiple image quality metrics (skew angle, blur/sharpness, contrast, illumination uniformity, and high-frequency noise).
2. **Selective Intervention**: If a page is already clean, it is passed through **untouched** to preserve original stroke geometry. If defects are detected, **only the specific corrective operations** required (e.g., deskew, background illumination normalization, CLAHE contrast enhancement) are applied.

---

## Pipeline Architecture

```
                 Input Document Page
                          │
                          ▼
        ┌───────────────────────────────────┐
        │       Quality Gate Assessment     │
        │  • Blur (Laplacian Variance)      │
        │  • Contrast (Std Dev)             │
        │  • Skew (Hough Transform Angle)   │
        │  • Illumination (Background Std)  │
        │  • Noise (High-Frequency Absdiff) │
        └─────────────────┬─────────────────┘
                          │
              Needs Preprocessing?
             /                    \
     [No]   /                      \   [Yes]
           ▼                        ▼
  Preserve Original        Selective Transformations
  (No Degradation)         ├── Deskew (Affine Rotation)
                           ├── Illumination Correction (Gaussian Division)
                           └── Contrast Enhancement (CLAHE)
           │                        │
           └───────────┬────────────┘
                       ▼
             Optimized OCR Image
```

---

## Quality Gate Metrics & Operations

The quality gate evaluates images using OpenCV and decides which corrective actions are needed:

| Metric | Measurement Technique | Threshold Condition | Triggered Operation | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Skew** | Canny edges + Probabilistic Hough Lines (`HoughLinesP`), filtering angles within $\pm 15^\circ$ | `abs(skew_angle) > 1.5°` | `deskew` | Corrects tilt using affine transformation while maintaining canvas dimensions and border replication. |
| **Illumination** | Downsampled Gaussian blur standard deviation | `illumination_score > 18.0` | `illumination` | Corrects non-uniform lighting / book fold gradients via background division. |
| **Noise** | Difference between raw grayscale and $3 \times 3$ Gaussian blur (`cv2.absdiff`) | `noise_score > 8.0` | `denoise` | Filters high-frequency background grain and scan artifacts. |
| **Blur** | Variance of the Laplacian (`cv2.Laplacian`) | `blur_score < 80.0` | `sharpen` | Detects soft focus or out-of-focus capture. |
| **Contrast** | Grayscale intensity standard deviation | `contrast_score < 35.0` | `contrast` | Enhances faded ink or low-contrast pages using CLAHE without blowing out stroke details. |

---

## Project Structure

```
.
├── input/                          # Sample input document pages (e.g., page001.png, page002.png)
├── output/                         # Processed and test outputs
├── preprocessing/                  # Core package
│   ├── __init__.py
│   ├── quality_gate.py             # Page assessment & decision engine
│   ├── kannada_preprocess.py       # Selective preprocessing & deskew implementation
│   ├── auto_preprocess.py          # End-to-end automated pipeline CLI
│   ├── contrast_test.py            # CLAHE contrast enhancement module
│   └── illumination_test.py        # Large-scale Gaussian illumination correction module
├── requirements.txt                # Python dependencies
├── .python-version                 # Supported Python version (3.10+)
└── README.md                       # Documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- `pip`

### Setup Environment

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

## Usage

### 1. Automated Quality-Aware Preprocessing (Recommended)

Run the end-to-end pipeline on any document image. The system assesses quality and applies only necessary corrections:

```bash
python -m preprocessing.auto_preprocess input/page001.png output/page001_auto.png
```

**Example 1: Page with Skew (`page001.png`)**
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

**Example 2: Clean Page (`page002.png`)**
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

### 2. Standalone Page Quality Assessment

Inspect quality metrics and suggested operations without altering the image:

```bash
python -m preprocessing.quality_gate path/to/page.png
```

---

### 3. Modular Testing & Experimentation

- **Deskew Processing**:
  ```bash
  python -m preprocessing.kannada_preprocess
  ```
- **Illumination Correction**:
  ```bash
  python -m preprocessing.illumination_test
  ```
- **Contrast Enhancement (CLAHE)**:
  ```bash
  python -m preprocessing.contrast_test
  ```

---

## Python API Integration

You can integrate the quality gate and preprocessing directly into your custom OCR or document ingestion workflow:

```python
import cv2
from preprocessing.quality_gate import assess_page
from preprocessing.kannada_preprocess import preprocess_page

# Load image
image = cv2.imread("input/page001.png")

# Run quality gate assessment
quality = assess_page(image)

print(f"Skew detected: {quality.skew_angle}°")
print(f"Recommended operations: {quality.operations}")

# Process selectively
if quality.needs_preprocessing:
    processed_image = preprocess_page(
        image,
        skew_angle=quality.skew_angle,
        operations=quality.operations,
    )
else:
    processed_image = image
```

---

## Key Design Principles

1. **Stroke Integrity First**: Kannada characters have distinct circular forms, upper marks, and subscript consonants. Heavy thresholding can easily merge or sever these strokes.
2. **Deterministic Thresholds**: Quality scores are continuous float metrics, allowing easy tuning against specific scanner types, camera captures, or historic print styles.
3. **Reproducible & Inspectable**: Every execution reports exact diagnostic scores before taking action, ensuring complete visibility into pipeline decisions.

---

## License

This project is licensed under the [MIT License](LICENSE) (or the repository's designated open-source license).
