import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.segmentation import threshold_segmentation
def test_threshold_segmentation():
    image = np.array(
        [
            [50, 100],
            [150, 200]
        ],
        dtype=np.uint8
    )
    mask = threshold_segmentation(
        image,
        threshold=100,
        rule="greater"
    )
    expected = np.array(
        [
            [0, 0],
            [1, 1]
        ],
        dtype=np.uint8
    )
    assert np.array_equal(mask, expected)
if __name__ == "__main__":
    test_threshold_segmentation()
    print("threshold segmentation test passed")