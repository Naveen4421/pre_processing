"""
Kannada Document OCR Image Preprocessing Module.

This module implements targeted, non-destructive image transformations designed
specifically for Indic scripts like Kannada, where character stroke integrity,
subscript consonants (ottu/vatlu), and vowel signs (matras) must be preserved.

Transformation Pipeline Order:
    OP_ORDER = ["orientation", "crop", "deskew", "illumination", "denoise", "contrast", "sharpen"]

Why this specific execution order is required:
1. orientation first:
   Gross page rotation (90/180/270 degrees, e.g. a book photographed sideways
   or upside down) must be corrected before every other step, since crop's
   document-quad detection and deskew's line-angle detection both assume the
   page is already close to upright.
2. crop second:
   Non-document backgrounds (table surfaces, cloth, shadows, camera margins)
   must be excised before geometry correction and illumination normalization,
   ensuring subsequent filters only analyze the actual page content.
3. deskew third:
   Geometric alignment must occur before spatial filtering or intensity
   transformations so that coordinate spaces and orientation are normalized.
4. illumination before contrast:
   Large-scale background lighting gradients and book-fold shadows must be
   flattened before local histogram equalization; otherwise, CLAHE amplifies
   the background illumination unevenness into dark/blown-out regions.
5. denoise before contrast:
   High-frequency scanner and paper grain must be smoothed (using an
   edge-preserving bilateral filter) prior to contrast enhancement; otherwise,
   CLAHE intensifies background noise.
6. sharpen last:
   High-frequency edge enhancement (unsharp masking) must run at the very end;
   running it earlier would amplify noise and contrast artifacts across
   subsequent stages.
"""

from __future__ import annotations

import cv2
import numpy as np

OP_ORDER: list[str] = [
    "orientation",
    "crop",
    "deskew",
    "illumination",
    "denoise",
    "contrast",
    "sharpen",
]


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel grayscale if it has 3 channels."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 polygon coordinates: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_document_quad(image: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    Detect the 4-corner polygon of a document page in a camera photo or scan.

    Returns (corners, area_ratio) or (None, 0.0) if no external background is found.
    Area ratio is the fraction of total frame occupied by the detected document.
    """
    if image is None or image.size == 0:
        return None, 0.0

    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return None, 0.0

    gray = _to_gray(image)
    total_area = float(h * w)

    scale = 800.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
    else:
        small = gray.copy()
        scale = 1.0

    sh, sw = small.shape
    small_area = float(sh * sw)

    blurred = cv2.GaussianBlur(small, (9, 9), 0)
    edges = cv2.Canny(blurred, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.20 * small_area:
            break
        # If candidate already covers >= 95% of the image, the frame is already the document
        if area >= 0.95 * small_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            corners = (approx / scale).reshape(4, 2)
            area_ratio = (area / scale**2) / total_area
            return corners, area_ratio

    return None, 0.0


def rotate_orientation(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Correct gross page rotation by a clockwise multiple of 90 degrees.

    Unlike `deskew` (which nudges a nearly-upright page by a few degrees),
    this handles a page captured sideways or upside down. `angle` is the
    clockwise rotation needed to make the page upright, as produced by
    `quality_gate.detect_orientation`. No-op for angle 0 or any value that
    isn't a multiple of 90.
    """
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


def crop(image: np.ndarray) -> np.ndarray:
    """
    Dynamically detect and crop the book page boundary, trimming external
    backgrounds (tables, cloth, scanner margins) using perspective correction.

    Safety: If no distinct external background is found, returns a copy of
    the original image untouched (no-op).
    """
    corners, ratio = find_document_quad(image)
    if corners is None or ratio >= 0.95 or ratio < 0.20:
        return image.copy()

    rect = order_points(corners.astype(np.float32))
    tl, tr, br, bl = rect

    # Calculate width of new flattened document
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Calculate height of new flattened document
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    if max_width < 50 or max_height < 50:
        return image.copy()

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by `angle` degrees around its center while keeping canvas size.

    Preserves color channels if provided (operates on 2D or 3D arrays).
    Uses clean white border padding to prevent dark margin smears.
    No-op if abs(angle) < 0.01 degrees.
    """
    if abs(angle) < 0.01:
        return image.copy()

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    border_val = (255, 255, 255) if len(image.shape) == 3 else 255

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )


def illumination(image: np.ndarray, sigma: float = 35.0) -> np.ndarray:
    """
    Gaussian-division background illumination normalization.

    Estimates low-frequency background variation using a wide Gaussian blur,
    normalizes the foreground against it, and scales back to [0, 255].
    Converts 3D images to grayscale.
    """
    gray = _to_gray(image)
    gray_f = gray.astype(np.float32)

    bg = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=sigma)
    bg = np.maximum(bg, 1.0)

    normalized = (gray_f / bg) * 255.0
    return np.clip(normalized, 0.0, 255.0).astype(np.uint8)


def denoise(image: np.ndarray, strength: int = 7) -> np.ndarray:
    """
    Edge-preserving smoothing using a bilateral filter.

    Unlike Gaussian filtering (which dissolves thin Kannada matras and ottu loops),
    bilateral filtering smooths uniform paper grain while retaining sharp edges.
    Converts 3D images to grayscale.
    """
    gray = _to_gray(image)
    return cv2.bilateralFilter(
        gray,
        d=strength,
        sigmaColor=strength * 5,
        sigmaSpace=strength * 5,
    )


def contrast(image: np.ndarray, clip: float = 2.5, tiles: int = 8) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Improves local contrast without blowing out thin character strokes.
    Converts 3D images to grayscale.
    """
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return clahe.apply(gray)


def sharpen(image: np.ndarray, amount: float = 0.8) -> np.ndarray:
    """
    Unsharp mask sharpening.

    Blurs a copy with Gaussian sigma=1.0 and blends with the original.
    Keeps amount <= 1.0 to avoid halo ringing on fine script glyphs.
    Converts 3D images to grayscale.
    """
    gray = _to_gray(image)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)



def preprocess_page(
    image: np.ndarray,
    skew_angle: float = 0.0,
    orientation_angle: int = 0,
    operations: list[str] | None = None,
) -> np.ndarray:
    """
    Apply requested ops in the FIXED safe order OP_ORDER, regardless of the
    order they appear in `operations`. Unknown op -> ValueError.
    """
    if operations is None:
        return image.copy()

    for op in operations:
        if op not in OP_ORDER:
            raise ValueError(
                f"Unknown operation: '{op}'. Supported operations: {OP_ORDER}"
            )

    result = image.copy()
    for op in OP_ORDER:
        if op not in operations:
            continue

        if op == "orientation":
            result = rotate_orientation(result, orientation_angle)
        elif op == "crop":
            result = crop(result)
        elif op == "deskew":
            result = deskew(result, skew_angle)
        elif op == "illumination":
            result = illumination(result)
        elif op == "denoise":
            result = denoise(result)
        elif op == "contrast":
            result = contrast(result)
        elif op == "sharpen":
            result = sharpen(result)

    return result
