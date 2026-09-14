from pathlib import Path

import cv2
import numpy as np


def correct_illumination(image):
    """
    Correct slow background illumination variation
    while preserving text details.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image.copy()

    # Estimate the large-scale background.
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=25,
    )

    # Normalize the page against the estimated background.
    normalized = cv2.divide(
        gray,
        background,
        scale=255,
    )

    return normalized


def main():
    input_path = Path(
        "output/page001_deskew.png"
    )

    output_path = Path(
        "output/page001_illumination.png"
    )

    image = cv2.imread(
        str(input_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {input_path}"
        )

    corrected = correct_illumination(image)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        corrected,
    )

    print()
    print("========== ILLUMINATION TEST ==========")
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Shape  : {image.shape} -> {corrected.shape}")
    print("=======================================")


if __name__ == "__main__":
    main()
