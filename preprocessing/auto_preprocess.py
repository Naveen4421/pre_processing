from pathlib import Path

import cv2

from preprocessing.quality_gate import assess_image_file
from preprocessing.kannada_preprocess import preprocess_page


def auto_preprocess(input_path, output_path):
    """
    Automatically preprocess a page based on its quality.

    Good pages are kept unchanged.
    Bad pages receive only the preprocessing operations
    selected by the quality gate.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    image = cv2.imread(str(input_path))

    if image is None:
        raise RuntimeError(
            f"Could not read image: {input_path}"
        )

    # Step 1: assess the original page.
    quality = assess_image_file(input_path)

    # Step 2: no problems -> preserve the original.
    if not quality.operations:
        processed = image.copy()

        print()
        print("========== AUTO PREPROCESS ==========")
        print(f"Input      : {input_path}")
        print("Decision   : KEEP ORIGINAL")
        print("Operations : NONE")
        print(f"Output     : {output_path}")
        print("=====================================")

    else:
        # Step 3: apply only required operations.
        processed = preprocess_page(
            image,
            skew_angle=quality.skew_angle,
            operations=quality.operations,
        )

        print()
        print("========== AUTO PREPROCESS ==========")
        print(f"Input      : {input_path}")
        print("Decision   : PREPROCESS")
        print(f"Operations : {quality.operations}")
        print(f"Output     : {output_path}")
        print("=====================================")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(str(output_path), processed):
        raise RuntimeError(
            f"Could not write output: {output_path}"
        )

    return quality


def main():
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python -m preprocessing.auto_preprocess "
            "<input> <output>"
        )
        raise SystemExit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    auto_preprocess(
        input_path,
        output_path,
    )


if __name__ == "__main__":
    main()
