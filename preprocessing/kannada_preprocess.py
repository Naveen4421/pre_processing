"""
Kannada Document OCR Image Preprocessing Module.

This module implements targeted, non-destructive image transformations designed
specifically for Indic scripts like Kannada, where character stroke integrity,
subscript consonants (ottu/vatlu), and vowel signs (matras) must be preserved.

Transformation Pipeline Order:
    OP_ORDER = ["deskew", "illumination", "denoise", "contrast", "sharpen"]

Why this specific execution order is required:
1. deskew first:
   Geometric alignment must occur before spatial filtering or intensity
   transformations so that coordinate spaces and orientation are normalized.
2. illumination before contrast:
   Large-scale background lighting gradients and book-fold shadows must be
   flattened before local histogram equalization; otherwise, CLAHE amplifies
   the background illumination unevenness into dark/blown-out regions.
3. denoise before contrast:
   High-frequency scanner and paper grain must be smoothed (using an
   edge-preserving bilateral filter) prior to contrast enhancement; otherwise,
   CLAHE intensifies background noise.
4. sharpen last:
   High-frequency edge enhancement (unsharp masking) must run at the very end;
   running it earlier would amplify noise and contrast artifacts across
   subsequent stages.
"""

from __future__ import annotations

import cv2
import numpy as np

OP_ORDER: list[str] = ["deskew", "illumination", "denoise", "contrast", "sharpen"]


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel grayscale if it has 3 channels."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by `angle` degrees around its center while keeping canvas size.

    Preserves color channels if provided (operates on 2D or 3D arrays).
    No-op if abs(angle) < 0.01 degrees.
    """
    if abs(angle) < 0.01:
        return image.copy()

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def illumination(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
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


def contrast(image: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Improves local contrast without blowing out thin character strokes.
    Converts 3D images to grayscale.
    """
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return clahe.apply(gray)


def sharpen(image: np.ndarray, amount: float = 0.5) -> np.ndarray:
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

        if op == "deskew":
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
