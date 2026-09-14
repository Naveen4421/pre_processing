import cv2
from pathlib import Path


def deskew(image, skew_angle):
    """
    Correct page skew while keeping the original image size.

    The detected skew angle is corrected in the opposite direction.
    """
    if abs(skew_angle) < 0.01:
        return image.copy()

    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # Correct the detected skew.
    # Positive angle = counter-clockwise
    # Negative angle = clockwise
    correction_angle = skew_angle

    matrix = cv2.getRotationMatrix2D(
        center,
        correction_angle,
        1.0,
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_page(image, skew_angle=0.0, operations=None):
    """
    Apply only the operations selected by the quality gate.
    """
    if operations is None:
        operations = []

    result = image.copy()

    if "deskew" in operations:
        result = deskew(result, skew_angle)

    return result


if __name__ == "__main__":
    from preprocessing.quality_gate import assess_image_file

    input_path = Path("input/page001.png")
    output_path = Path("output/page001_deskew.png")

    image = cv2.imread(str(input_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    quality = assess_image_file(input_path)

    processed = preprocess_page(
        image,
        skew_angle=quality.skew_angle,
        operations=quality.operations,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), processed)

    print()
    print("========== PREPROCESSING ==========")
    print(f"Input      : {input_path}")
    print(f"Operations : {quality.operations}")
    print(f"Skew       : {quality.skew_angle:.2f}°")
    print(f"Output     : {output_path}")
    print(f"Shape      : {image.shape} -> {processed.shape}")
    print("===================================")
