import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import calculate_iou, calculate_dice
def test_calculate_scores():
    predicted = np.array([
        [1, 1],
        [0, 1]
    ])
    ground_truth = np.array([
        [1, 0],
        [0, 1]
    ])
    score = calculate_iou(predicted, ground_truth)
    assert np.isclose(score, 2 / 3)
    dice_score = calculate_dice(predicted, ground_truth)
    assert np.isclose(dice_score, 0.8)

def test_empty_masks():
    empty = np.zeros((2, 2), dtype=np.uint8)
    assert calculate_iou(empty, empty) == 1.0
    assert calculate_dice(empty, empty) == 1.0

def test_mismatched_mask_shapes():
    predicted = np.zeros((2, 2), dtype=np.uint8)
    ground_truth = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Masks must have the same shape"):
        calculate_iou(predicted, ground_truth)
    with pytest.raises(ValueError, match="Masks must have the same shape"):
        calculate_dice(predicted, ground_truth)

if __name__ == "__main__":
    test_calculate_scores()
    test_empty_masks()
    test_mismatched_mask_shapes()
    print("metrics tests passed")