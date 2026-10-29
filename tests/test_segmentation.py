import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.segmentation import threshold_segmentation
from src.segmentation import kmeans_segmentation
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

def test_kmeans_segmentation():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    # create two different colour regions
    image[:, :10] = [220, 40, 40]
    image[:, 10:] = [40, 40, 220]
    mask = kmeans_segmentation(
        image,
        k=2,
        rule="highest_saturation",
        seed=31
    )
    assert mask.shape == (20, 20)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    assert np.count_nonzero(mask) > 0

if __name__ == "__main__":
    test_threshold_segmentation()
    test_kmeans_segmentation()
    print("segmentation tests passed")