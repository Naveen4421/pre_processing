from __future__ import annotations

from dataclasses import dataclass, asdict
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PageQuality:
    width: int
    height: int

    blur_score: float
    contrast_score: float
    skew_angle: float
    illumination_score: float
    noise_score: float
    orientation_angle: int

    needs_preprocessing: bool
    operations: list[str]

    def to_dict(self):
        return asdict(self)


def calculate_blur_score(gray: np.ndarray) -> float:
    """
    Higher value = sharper image.
    Lower value = more blur.
    """
    return float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )


def calculate_contrast_score(gray: np.ndarray) -> float:
    """
    Higher value generally means stronger contrast.
    """
    return float(gray.std())


def detect_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate page/text skew in degrees.
    """

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=max(gray.shape) // 5,
        maxLineGap=20,
    )

    if lines is None:
        return 0.0

    angles = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(-1)

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue

        angle = np.degrees(
            np.arctan2(dy, dx)
        )

        # Text/page lines are normally close
        # to horizontal.
        if -15 <= angle <= 15:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(
        np.median(angles)
    )


def detect_orientation(gray: np.ndarray) -> int:
    """
    Detect gross page rotation (0/90/180/270 degrees) via Tesseract's
    Orientation & Script Detection (OSD) mode.

    OSD only reports geometric orientation metadata; it does not perform
    text recognition. Returns the clockwise rotation (in degrees) needed to
    make the page upright, or 0 if tesseract is unavailable or orientation
    cannot be determined confidently (e.g. too little text on the page).
    """
    tess = shutil.which("tesseract")
    if not tess:
        return 0

    with tempfile.TemporaryDirectory(prefix="osd_") as td:
        img_path = Path(td) / "page.png"
        if not cv2.imwrite(str(img_path), gray):
            return 0

        res = subprocess.run(
            [tess, str(img_path), "stdout", "--psm", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode != 0:
            return 0

        for line in res.stdout.splitlines():
            if line.startswith("Rotate:"):
                try:
                    rotate = int(line.split(":")[1].strip())
                except ValueError:
                    return 0
                if rotate in (0, 90, 180, 270):
                    return rotate
                return 0

    return 0


def calculate_illumination_score(
    gray: np.ndarray,
) -> float:
    """
    Higher value = more uneven illumination.
    """

    small = cv2.resize(
        gray,
        (0, 0),
        fx=0.1,
        fy=0.1,
    )

    background = cv2.GaussianBlur(
        small,
        (0, 0),
        sigmaX=15,
    )

    return float(
        background.std()
    )


def calculate_noise_score(
    gray: np.ndarray,
) -> float:
    """
    Rough high-frequency noise estimate.
    Higher value = more noise.
    """

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    noise = cv2.absdiff(
        gray,
        blurred,
    )

    return float(
        noise.mean()
    )


def decide_operations(
    blur_score: float,
    contrast_score: float,
    skew_angle: float,
    illumination_score: float,
    noise_score: float,
    needs_crop: bool = False,
    orientation_angle: int = 0,
) -> list[str]:

    operations = []

    # Gross page rotation (90/180/270) must be corrected before crop/deskew,
    # which only handle fine geometry.
    if orientation_angle:
        operations.append("orientation")

    # If camera capture includes foreign background (table, cloth), crop first
    if needs_crop:
        operations.append("crop")

    # Tuned thresholds for real Kannada document & book scans.
    # Correct even slight skew (as low as ~1 degree) so text baselines stay level.
    if abs(skew_angle) > 0.5:
        operations.append("deskew")

    if illumination_score > 18:
        operations.append("illumination")
        # Illumination division normalizes lighting but requires contrast
        # enhancement to restore deep character ink density
        if "contrast" not in operations:
            operations.append("contrast")

    if noise_score > 8:
        operations.append("denoise")

    # Document text scans with blur_score < 600 exhibit optical softness/blur
    if blur_score < 600:
        operations.append("sharpen")

    if contrast_score < 35:
        if "contrast" not in operations:
            operations.append("contrast")

    return operations


def assess_page(
    image: np.ndarray,
) -> PageQuality:

    if image is None:
        raise ValueError(
            "Image is None"
        )

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image.copy()

    height, width = gray.shape

    # Detect if image is a camera photo containing non-document background
    try:
        from preprocessing.kannada_preprocess import find_document_quad
        corners, ratio = find_document_quad(image)
        needs_crop = bool(corners is not None and 0.20 <= ratio < 0.95)
    except Exception:
        needs_crop = False

    blur_score = calculate_blur_score(
        gray
    )

    contrast_score = calculate_contrast_score(
        gray
    )

    skew_angle = detect_skew_angle(
        gray
    )

    illumination_score = (
        calculate_illumination_score(
            gray
        )
    )

    noise_score = calculate_noise_score(
        gray
    )

    orientation_angle = detect_orientation(
        gray
    )

    operations = decide_operations(
        blur_score=blur_score,
        contrast_score=contrast_score,
        skew_angle=skew_angle,
        illumination_score=illumination_score,
        noise_score=noise_score,
        needs_crop=needs_crop,
        orientation_angle=orientation_angle,
    )

    return PageQuality(

        width=width,
        height=height,

        blur_score=round(
            blur_score,
            2
        ),

        contrast_score=round(
            contrast_score,
            2
        ),

        skew_angle=round(
            skew_angle,
            2
        ),

        illumination_score=round(
            illumination_score,
            2
        ),

        noise_score=round(
            noise_score,
            2
        ),

        orientation_angle=orientation_angle,

        needs_preprocessing=(
            len(operations) > 0
        ),

        operations=operations,
    )


def assess_image_file(
    image_path: str,
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read image: "
            f"{image_path}"
        )

    result = assess_page(
        image
    )

    print()
    print(
        "========== PAGE QUALITY =========="
    )

    print(
        f"Image        : {image_path}"
    )

    print(
        f"Resolution   : "
        f"{result.width} x {result.height}"
    )

    print(
        f"Blur         : "
        f"{result.blur_score}"
    )

    print(
        f"Contrast     : "
        f"{result.contrast_score}"
    )

    print(
        f"Skew         : "
        f"{result.skew_angle}°"
    )

    print(
        f"Orientation  : "
        f"{result.orientation_angle}° (rotation needed)"
    )

    print(
        f"Illumination : "
        f"{result.illumination_score}"
    )

    print(
        f"Noise        : "
        f"{result.noise_score}"
    )

    print()

    print(
        "Needs preprocessing: "
        f"{result.needs_preprocessing}"
    )

    if result.operations:

        print("Operations:")

        for operation in result.operations:
            print(
                f"  - {operation}"
            )

    else:

        print(
            "Operations: NONE"
        )

    print(
        "=================================="
    )

    print()

    return result


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Assess image quality "
            "before OCR preprocessing"
        )
    )

    parser.add_argument(
        "image",
        help="Path to image"
    )

    args = parser.parse_args()

    assess_image_file(
        args.image
    )
