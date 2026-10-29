import numpy as np
def calculate_iou(predicted, ground_truth):
    # IoU Score
    predicted = np.asarray(predicted) > 0
    ground_truth = np.asarray(ground_truth) > 0
    intersection = np.logical_and(predicted, ground_truth).sum()
    union = np.logical_or(predicted, ground_truth).sum()
    if union == 0:
        return 1.0
    return intersection / union

def calculate_dice(predicted, ground_truth):
    # Dice Score
    predicted = np.asarray(predicted) > 0
    ground_truth = np.asarray(ground_truth) > 0
    intersection = np.logical_and(predicted, ground_truth).sum()
    total = predicted.sum() + ground_truth.sum()
    if total == 0:
        return 1.0
    return (2 * intersection) / total
