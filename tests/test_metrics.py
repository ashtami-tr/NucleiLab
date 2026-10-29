import sys
from pathlib import Path
import numpy as np
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
if __name__ == "__main__":
    test_calculate_scores()
    print("metrics tests passed")