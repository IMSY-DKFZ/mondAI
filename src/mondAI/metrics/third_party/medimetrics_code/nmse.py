# mypy: ignore-errors
# Source: https://github.com/Bayer-Group/mr-image-metrics/blob/main/medimetrics/metrics/nmse.py
# Commit a70fe67843b954a9dbeb4b615eb6952df282d909
# License: BSD 3-Clause License
from typing import Any

import numpy as np

from .base import FullRefMetric


class NMSE(FullRefMetric):
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image_true: np.array (H, W)
        Reference image
    image_test: np.array (H, W)
        Image to be evaluated against the reference image
    """

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, **kwargs: Any) -> float:
        mse = np.power(image_true - image_test, 2).mean()

        # torch.std corrects the std dev by default, so do the same here!
        stddev = np.std(image_true, ddof=1)

        if stddev > 0:
            return mse / stddev
        else:
            return np.inf
