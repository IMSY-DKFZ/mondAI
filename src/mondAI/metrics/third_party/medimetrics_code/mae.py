# mypy: ignore-errors
# Source: https://github.com/Bayer-Group/mr-image-metrics/blob/main/medimetrics/metrics/mse.py
# Commit a70fe67843b954a9dbeb4b615eb6952df282d909
# License: BSD 3-Clause License
from typing import Any

import numpy as np

from .base import FullRefMetric


class MAE(FullRefMetric):
    def __init__(self) -> None:
        pass

    """
    Parameters:
    -----------
    image_true: np.array (H, W)
        Reference image
    image_test: np.array (H, W)
        Image to be evaluated against the reference image
    data_range:
        By default use joint maximum - joint minimum
    """

    def compute(self, image_true: np.ndarray, image_test: np.ndarray, **kwargs: Any) -> float:
        return np.mean(np.abs(image_true - image_test))
