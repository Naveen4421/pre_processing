from pathlib import Path

import cv2
import numpy as np


def enhance_contrast(image):
    """
    Improve local contrast while preserving Kannada character strokes.

    CLAHE is used instead of aggressive global contrast stretching.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    return enhanced


def main():
    input_path = Path("output/page001_auto.png")
    output_path = Path("output/page001_contrast.png")

    image = cv2.imread(str(input_path))

    if image is None:
        raise RuntimeError(
            f"Could not read image: {input_path}"
        )

    enhanced = enhance_contrast(image)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        enhanced,
    )

    print()
    print("========== CONTRAST TEST ==========")
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Shape  : {image.shape} -> {enhanced.shape}")
    print("===================================")


if __name__ == "__main__":
    main()
