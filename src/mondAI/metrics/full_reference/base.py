from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch

from mondAI.metrics.base import Metric
from mondAI.metrics.dimension import _DIMENSION_LOOKUP
from mondAI.utils.checks import (
    check_dimensions,
    check_image_type,
    check_nan_values,
    check_same_shape,
    check_same_type,
)
from mondAI.utils.internal_format import convert_to_internal_format


class FullReferenceMetric(Metric, ABC):
    """Abstract base class for metrics that require a reference image."""

    def __call__(
        self,
        image: np.ndarray | torch.Tensor,
        reference: np.ndarray | torch.Tensor,
        dims: Sequence[str] = ("W", "H"),
    ) -> float | torch.Tensor | np.ndarray:
        """Compute the metric between image and corresponding reference."""
        # Check inputs
        check_image_type(image)
        check_image_type(reference)
        check_same_type(image, reference)
        check_same_shape(image, reference)
        check_nan_values(image, reference=False)
        check_nan_values(reference, reference=True)
        check_dimensions(dims, image)

        # Convert to torch tensors if they are numpy arrays for easier processing, save original type and torch device
        # to be able to convert back for consistency in output type. This will allow us to leverage PyTorch's efficient
        # tensor operations while maintaining compatibility with both input types.
        original_type = type(image)
        if original_type == torch.Tensor:
            original_torch_device = image.device

        image = convert_to_internal_format(image)
        reference = convert_to_internal_format(reference)

        self._check_metric_configuration()

        # converts dimension strings to Dimension enums
        dims_enum = [_DIMENSION_LOOKUP[dim] for dim in dims]

        # check that image has the expected dimensions for the metric
        for expected_dim in self.expected_dimensions:
            if expected_dim not in dims_enum:
                raise ValueError(f"Expected dimension '{expected_dim}' not found in specified dimensions {dims_enum}.")

        expected_dim_indices = [dims_enum.index(dim) for dim in self.expected_dimensions]
        other_dim_indices = [i for i in range(image.ndim) if i not in expected_dim_indices]

        # Permute dimensions to expected order
        permute_order = other_dim_indices + expected_dim_indices
        image_permuted = image.permute(permute_order)
        reference_permuted = reference.permute(permute_order)

        metric_shape = image_permuted.shape[len(other_dim_indices) :]
        other_shape = image_permuted.shape[: len(other_dim_indices)]

        # reshape so that vmap iterates over "other dimensions"
        image_view = image_permuted.view(-1, *metric_shape)
        reference_view = reference_permuted.view(-1, *metric_shape)

        # Compute metric for each pair of images in the batch
        scores = torch.vmap(self._compute)(image_view, reference_view)

        # reshape back to original "other dimensions" shape if necessary, otherwise return scalar
        if len(other_shape) > 0:
            scores = scores.reshape(other_shape)
        else:
            return scores.item()

        # convert to numpy if necessary for consistency with input types
        if original_type == torch.Tensor:
            scores = scores.to(original_torch_device)
        else:
            scores = scores.cpu().numpy()

        return scores

    @abstractmethod
    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> float | torch.Tensor:
        """Internal method to compute the metric.

        To be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses should implement this method.")
