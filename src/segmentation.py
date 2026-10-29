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

def kmeans_segmentation(image, k=2, rule="highest_saturation", seed=31):
    # Group image pixels by colour and select a foreground cluster
    if image is None or image.size == 0:
        raise ValueError("Please provide a valid image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be an RGB image.")
    if not isinstance(k, int) or not 2 <= k <= 5:
        raise ValueError("k must be an integer between 2 and 5.")
    if rule not in ("darkest", "highest_saturation"):
        raise ValueError("Unknown cluster selection rule.")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(seed)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.float32)
    if rule == "darkest":
        selected_cluster = np.argmin(centers[:, 2])
    else:
        selected_cluster = np.argmax(centers[:, 1])
    mask = (labels.flatten() == selected_cluster)
    return mask.reshape(image.shape[:2]).astype(np.uint8)