import cv2
import numpy as np
def threshold_segmentation(image, threshold=127, rule="greater"):
    # Segment an image using grayscale pixel intensity
    if image is None or image.size == 0:
        raise ValueError("Please provide a valid image.")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image.copy()
    else:
        raise ValueError("Image must be grayscale or RGB.")
    if not 0 <= threshold <= 255:
        raise ValueError("Threshold must be between 0 and 255.")
    if rule == "greater":
        mask = gray > threshold
    elif rule == "less":
        mask = gray < threshold
    else:
        raise ValueError("Rule must be 'greater' or 'less'.")
    return mask.astype(np.uint8)
