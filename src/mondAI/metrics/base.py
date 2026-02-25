from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

import numpy as np
import torch

from mondAI.settings import settings
from mondAI.utils.checks import check_dimensions, check_image_type, check_nan_values
from mondAI.utils.internal_format import convert_to_internal_format

from .dimension import _DIMENSION_LOOKUP, Dimension


class Metric(ABC):
    """Abstract base class for all metrics."""

    name: str
    abbreviation: str
    higher_is_better: bool

    expected_dimensions: tuple[Dimension, ...]

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> float | list[float]:
        """Compute the metric score."""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def __str__(self) -> str:
        """Full text representation of the metric including its name and abbreviation
        and parameters for reproducible reporting."""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def fingerprint(self) -> dict[str, Any]:
        """Return a dictionary that uniquely identifies the metric and its parameters
        for reproducibility."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _arrow_indicating_optimum(self) -> str:
        """Show an arrow indicating whether higher metric values are better.

        :return: "↑" if higher is better, "↓" otherwise.
        :rtype: str

        """
        return "↑" if self.higher_is_better else "↓"

    @abstractmethod
    def _compute_vmapped(
        self,
        *views: torch.Tensor,
        compute_function: Callable[..., float | torch.Tensor | np.ndarray],
    ) -> torch.Tensor:
        """Vectorized compute function over flattened items.

        Must return a 1D tensor (N,).

        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _call_with_vectorization(
        self,
        *inputs: np.ndarray | torch.Tensor,
        dims: Sequence[str] = settings.default_dims,
        compute_function: Callable[..., float | torch.Tensor | np.ndarray],
    ) -> float | torch.Tensor | np.ndarray:
        """Shared call pipeline for metrics, including:

        - common input checks (type/nan/dims)
        - conversion to internal torch representation
        - expected-dimension validation + permutation/flattening
        - vectorized execution (subclass hook)
        - output restoration

        :param inputs: One or more input images (e.g. image and reference) as numpy arrays or torch tensors.
        :type inputs: tuple[np.ndarray | torch.Tensor, ...]
        :param dims: The dimensions of the input images, specified as a sequence of strings.
        :type dims: Sequence[str]
        :param compute_function: The function to compute the metric, which will be vectorized.
        :type compute_function: Callable[..., float | torch.Tensor | np.ndarray]
        :return: The computed metric score, as a scalar or array/tensor depending on the input shape and type.
        :rtype: float | torch.Tensor | np.ndarray
        :raises ValueError: If the number of inputs is not 1 or 2, if the input images contain NaN values, or if the
        expected dimensions are not present in the specified dimensions.

        """

        # checks on number of inputs, e.g. 1 for no-reference metrics, 2 for full-reference metrics
        if len(inputs) not in (1, 2):
            raise ValueError(f"{type(self).__name__} supports 1 or 2 input images, got {len(inputs)}.")

        # checks input images
        for i, image in enumerate(inputs):
            check_image_type(image)
            check_nan_values(image, reference=(i == 1))  # assumes that reference is second image
        check_dimensions(dims, inputs[0])

        # Convert to torch tensors if they are numpy arrays for easier processing, save original type and torch device
        # to be able to convert back for consistency in output type. This will allow us to leverage PyTorch's efficient
        # tensor operations while maintaining compatibility with both input types.
        output_is_torch = isinstance(inputs[0], torch.Tensor)
        output_device = inputs[0].device if output_is_torch else None

        torch_inputs = tuple(convert_to_internal_format(x) for x in inputs)

        dims_enum = [_DIMENSION_LOOKUP[d] for d in dims]

        for expected_dim in self.expected_dimensions:
            if expected_dim not in dims_enum:
                raise ValueError(f"Expected dimension '{expected_dim}' not found in specified dimensions {dims_enum}.")

        expected_dim_indices = [dims_enum.index(dim) for dim in self.expected_dimensions]
        other_dim_indices = [i for i in range(torch_inputs[0].ndim) if i not in expected_dim_indices]

        reshaped_images: list[torch.Tensor] = []

        # Permute dimensions to expected order
        for image in torch_inputs:
            permute_order = other_dim_indices + expected_dim_indices
            image_permuted = image.permute(permute_order)

            metric_shape = image_permuted.shape[len(other_dim_indices) :]
            other_shape = image_permuted.shape[: len(other_dim_indices)]

            # reshape so that vmap iterates over "other dimensions"
            image_reshaped = image_permuted.reshape(-1, *metric_shape)
            reshaped_images.append(image_reshaped)

        # Compute metric for each image (pair)
        scores_1d = self._compute_vmapped(*reshaped_images, compute_function=compute_function)

        # return scalar if there are no "other dimensions" to iterate over
        if len(other_shape) == 0:
            return scores_1d.item()

        # reshape back to original "other dimensions" shape
        scores = scores_1d.reshape(other_shape)

        # convert to numpy if necessary for consistency with input types
        if output_is_torch:
            return scores.to(output_device) if output_device is not None else scores
        return scores.detach().cpu().numpy()


class MetricList:
    """A list of metrics."""

    def __init__(self, list_name: str = "", metrics: list[Metric] | None = None) -> None:
        self.list_name = list_name
        self.metrics = metrics if metrics is not None else []

    def __call__(self, *args: Any, **kwargs: Any) -> list[float | list[float]]:
        return [m(*args, **kwargs) for m in self.metrics]

    def __str__(self) -> str:
        return f"MetricList(name={self.list_name}, metrics={[str(m) for m in self.metrics]})"

    def fingerprint(self) -> dict[str, Any]:
        return {
            "list_name": self.list_name,
            "metrics": [m.fingerprint() for m in self.metrics],
        }
