"""This file contains fixtures for testing the metrics defined in `metrics`. These
fixtures provide a variety of test images and volumes, including the Shepp-Logan
phantom and a brain volume, as well as random images of different shapes and
dimensions. Additionally, it includes dummy implementations of a full reference metric
and a no reference metric for testing purposes.

These fixtures can be used across multiple test files to ensure consistency and reduce
code duplication when testing the metrics' functionality and behavior on different
types of input data.

"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytest import fixture
from skimage.data import brain as brain_volume
from skimage.data import shepp_logan_phantom

from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.no_reference.base import NoReferenceMetric


def load_shepp_logan_phantom() -> np.ndarray:
    # Load the Shepp-Logan phantom (400, 400) pixel image
    return shepp_logan_phantom()


def load_brain() -> np.ndarray:
    # Load the brain volume (10, 256, 256) voxel image
    # TODO: Should this volume be normalized to [0, 1]?
    # TODO: Does this need to be converted from uint8 to float64?
    return brain_volume()


def create_random_image(shape: tuple[int, ...], dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Create a random image tensor with the specified shape and data type."""
    return torch.rand(shape, dtype=dtype)


@fixture
def phantom() -> np.ndarray:
    return load_shepp_logan_phantom()


@fixture
def brain() -> np.ndarray:
    return load_brain()


@fixture
def brain_slice() -> np.ndarray:
    return load_brain()[4]  # Return the middle slice of the brain volume


@fixture
def grayscale_image_W_H() -> torch.Tensor:
    return create_random_image((256, 256))


@fixture
def rgb_image_C_W_H() -> torch.Tensor:
    return create_random_image((3, 256, 256))


@fixture
def rgb_image_W_H_C() -> torch.Tensor:
    return create_random_image((256, 256, 3))


@fixture
def dummy_full_reference_metric() -> FullReferenceMetric:
    class DummyFullReferenceMetric(FullReferenceMetric):
        name = "Dummy Full Reference Metric"
        abbreviation = "DFRM"
        higher_is_better = True

        def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            return torch.Tensor(0.0)

        def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
            return {}

        def __str__(self) -> str:
            return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"

        def fingerprint(self) -> dict[str, str | bool]:
            return {
                "name": self.name,
                "abbreviation": self.abbreviation,
                "higher_is_better": self.higher_is_better,
            }

    return DummyFullReferenceMetric()


@fixture
def dummy_no_reference_metric() -> NoReferenceMetric:
    class DummyNoReferenceMetric(NoReferenceMetric):
        name = "Dummy No Reference Metric"
        abbreviation = "DNRM"
        higher_is_better = True

        def _compute(self, image: torch.Tensor) -> torch.Tensor:
            return torch.Tensor(0.0)

        def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
            return {}

        def __str__(self) -> str:
            return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"

        def fingerprint(self) -> dict[str, str | bool]:
            return {
                "name": self.name,
                "abbreviation": self.abbreviation,
                "higher_is_better": self.higher_is_better,
            }

    return DummyNoReferenceMetric()


@fixture
def output_shape_test_metric_factory() -> Callable[[tuple[Dimension, ...]], FullReferenceMetric]:
    def _make_metric(metric_dimensions: tuple[Dimension, ...]) -> FullReferenceMetric:
        class OutputShapeTestMetric(FullReferenceMetric):
            name = "Output Shape Test Metric"
            abbreviation = "OSTM"
            higher_is_better = True

            expected_dimensions = metric_dimensions

            def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                return torch.mean(image)  # dummy computation

            def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
                return {}

            def __str__(self) -> str:
                return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"

            def fingerprint(self) -> dict[str, str | bool]:
                return {
                    "name": self.name,
                    "abbreviation": self.abbreviation,
                    "higher_is_better": self.higher_is_better,
                }

        return OutputShapeTestMetric()

    return _make_metric


# Example usage
if __name__ == "__main__":
    phantom = load_shepp_logan_phantom()
    print("Shepp-Logan Phantom shape:", phantom.shape, type(phantom))

    brain = load_brain()
    print("Brain Volume shape:", brain.shape, type(brain))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Shepp-Logan Phantom")
    plt.imshow(phantom, cmap="gray")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Brain")
    plt.imshow(brain[4], cmap="gray")
    plt.colorbar()
    plt.show()
