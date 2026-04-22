"""This file contains fixtures for testing the metrics defined in `metrics`. These
fixtures provide a variety of test images and volumes, including the Shepp-Logan
phantom and a brain volume, as well as random images of different shapes and
dimensions. Additionally, it includes dummy implementations of a full reference metric
and a no reference metric for testing purposes.

These fixtures can be used across multiple test files to ensure consistency and reduce
code duplication when testing the metrics' functionality and behavior on different
types of input data.

"""

from collections.abc import Callable

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
    # Load the Shepp-Logan phantom (400, 400) pixel image with values in [0, 255]
    phantom = shepp_logan_phantom()
    return (phantom - phantom.min()) / (phantom.max() - phantom.min()) * 255.0


def load_brain() -> np.ndarray:
    # Load the brain volume (10, 256, 256) voxel image
    # TODO: Should this volume be normalized to [0, 1]?
    # TODO: Does this need to be converted from uint8 to float64?
    brain = brain_volume()
    return (brain - brain.min()) / (brain.max() - brain.min()) * 255.0


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
def maximum_dimensions_image_B_C_D_H_W() -> np.ndarray:
    phantom = load_shepp_logan_phantom() / 255.0  # Normalize to [0, 1] for testing
    # Create a 5D image with dimensions (B, C, D, H, W) by repeating the phantom across new dimensions
    return np.tile(phantom, (5, 3, 10, 1, 1))  # (B=5, C=3, D=10, H=400, W=400)


@fixture
def grayscale_image_W_H() -> torch.Tensor:
    return create_random_image((256, 256))


@fixture
def grayscale_image_odd_W_H() -> torch.Tensor:
    return create_random_image((257, 257))


@fixture
def rgb_image_C_W_H() -> torch.Tensor:
    return create_random_image((3, 256, 256))


@fixture
def rgb_image_odd_C_W_H() -> torch.Tensor:
    return create_random_image((3, 256, 256))


@fixture
def rgb_image_W_H_C() -> torch.Tensor:
    return create_random_image((256, 256, 3))


@fixture
def dummy_full_reference_metric() -> FullReferenceMetric:
    class DummyFullReferenceMetric(FullReferenceMetric):
        @property
        def name(self) -> str:
            return "Dummy Full Reference Metric"

        @property
        def abbreviation(self) -> str:
            return "DFRM"

        @property
        def higher_is_better(self) -> bool:
            return True

        @property
        def expected_dimensions(self) -> tuple[Dimension, ...]:
            return (Dimension.HEIGHT, Dimension.WIDTH)

        def __init__(self) -> None:
            super().__init__()

        def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            return torch.Tensor(0.0)

        def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
            return {}

        def __str__(self) -> str:
            return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"

    return DummyFullReferenceMetric()


@fixture
def dummy_no_reference_metric() -> NoReferenceMetric:
    class DummyNoReferenceMetric(NoReferenceMetric):
        @property
        def name(self) -> str:
            return "Dummy Full Reference Metric"

        @property
        def abbreviation(self) -> str:
            return "DFRM"

        @property
        def higher_is_better(self) -> bool:
            return True

        @property
        def expected_dimensions(self) -> tuple[Dimension, ...]:
            return (Dimension.HEIGHT, Dimension.WIDTH)

        def _compute(self, image: torch.Tensor) -> torch.Tensor:
            return torch.Tensor(0.0)

        def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
            return {}

        def __str__(self) -> str:
            return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"

    return DummyNoReferenceMetric()


@fixture
def output_shape_test_metric_factory() -> Callable[[tuple[Dimension, ...]], FullReferenceMetric]:
    def _make_metric(metric_dimensions: tuple[Dimension, ...]) -> FullReferenceMetric:
        class OutputShapeTestMetric(FullReferenceMetric):
            @property
            def name(self) -> str:
                return "Output Shape Test Metric"

            @property
            def abbreviation(self) -> str:
                return "OSTM"

            @property
            def higher_is_better(self) -> bool:
                return True

            @property
            def expected_dimensions(self) -> tuple[Dimension, ...]:
                return metric_dimensions

            def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                return torch.mean(image)  # dummy computation

            def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
                return {}

            def __str__(self) -> str:
                return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"

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
