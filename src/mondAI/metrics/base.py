import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from mondAI.logging import get_logger
from mondAI.settings import settings
from mondAI.utils.checks import check_dimensions, check_image_type, check_nan_values
from mondAI.utils.internal_format import convert_to_internal_format

from .dimension import _DIMENSION_LOOKUP, Dimension

logger = get_logger()


class Metric(ABC):
    """Abstract base class for all metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Full name of the metric for reporting."""
        ...

    @property
    @abstractmethod
    def abbreviation(self) -> str:
        """Short abbreviation of the metric for compact reporting."""
        ...

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """Whether higher metric values indicate better quality."""
        ...

    @property
    @abstractmethod
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        """The dimensions that the metric expects to be present in the input images.

        This should be a tuple of Dimension enums indicating which dimensions are relevant for the metric computation.
        For example, a metric that operates on 2D spatial dimensions might expect (Dimension.HEIGHT, Dimension.WIDTH).

        :return: A tuple of Dimension enums indicating the expected dimensions for the metric.
        :rtype: tuple[Dimension, ...]

        """
        ...

    @abstractmethod
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> float | torch.Tensor | np.ndarray | dict[str, float | torch.Tensor | np.ndarray]:
        """Compute the metric score."""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def __str__(self) -> str:
        """Full text representation of the metric including its name and abbreviation
        and parameters for reproducible reporting."""
        raise NotImplementedError("Subclasses should implement this method.")

    def fingerprint(self) -> dict[str, Any]:
        """Return a dictionary that uniquely identifies the metric and its parameters
        for reproducibility.

        :return: A dictionary that uniquely identifies the metric and its parameters
            for reproducibility.
        :rtype: dict[str, Any]

        """

        signature = inspect.signature(self.__class__.__init__)
        params = {name: getattr(self, name) for name in signature.parameters if name != "self"}

        return {
            "metric": type(self).__name__,
            "name": self.name,
            "abbreviation": self.abbreviation,
            "higher_is_better": self.higher_is_better,
            "expected_dimensions": [dim.value for dim in self.expected_dimensions],
            **params,
        }

    def _arrow_indicating_optimum(self) -> str:
        """Show an arrow indicating whether higher metric values are better.

        :return: "↑" if higher is better, "↓" otherwise.
        :rtype: str

        """
        return "↑" if self.higher_is_better else "↓"

    @abstractmethod
    def _compute_iteratively(
        self,
        *views: torch.Tensor,
        compute_function: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        """Iteratively compute function over flattened items.

        Must return a 1D tensor (N,).

        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _call_pipeline(
        self,
        *inputs: np.ndarray | torch.Tensor,
        dims: Sequence[str] = settings.default_dims,
        compute_function: Callable[..., torch.Tensor],
    ) -> float | torch.Tensor | np.ndarray:
        """Shared call pipeline for metrics, including:

        - common input checks (type/nan/dims)
        - conversion to internal torch representation
        - expected-dimension validation + permutation/flattening
        - iterative execution (subclass hook)
        - output restoration

        :param inputs: One or more input images (e.g. image and reference) as numpy arrays or torch tensors.
        :type inputs: tuple[np.ndarray | torch.Tensor, ...]
        :param dims: The dimensions of the input images, specified as a sequence of strings.
        :type dims: Sequence[str]
        :param compute_function: The function to compute the metric, which will be iteratively applied.
        :type compute_function: Callable[..., torch.Tensor]
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
        logger.debug(f"Input was {'torch.Tensor' if output_is_torch else 'numpy.ndarray'} on device {output_device}.")

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

            # reshape so that `_compute_iteratively` iterates over "other dimensions"
            image_reshaped = image_permuted.reshape(-1, *metric_shape)
            reshaped_images.append(image_reshaped)

        logger.debug(f"Metric shape: {metric_shape}, other shape: {other_shape}, permute order: {permute_order}")

        # Compute metric for each image (pair)
        scores_1d = self._compute_iteratively(*reshaped_images, compute_function=compute_function)

        # return scalar if there are no "other dimensions" to iterate over
        if len(other_shape) == 0:
            return scores_1d.item()

        # reshape back to original "other dimensions" shape
        scores = scores_1d.reshape(other_shape)

        # convert to numpy if necessary for consistency with input types
        if output_is_torch:
            logger.debug(f"Converting output to original torch device {output_device}.")
            return scores.to(output_device) if output_device is not None else scores

        logger.debug("Converting output to numpy array for consistency with input type.")
        return scores.detach().cpu().numpy()

    @abstractmethod
    def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
        """Return a dictionary of other implementations of the metric for comparison.

        The keys should be the names of the libraries or implementations, and the values
        should be callables that compute the metric with the same signature as the main
        implementation.

        If no peer implementations are available return an empty dictionary.

        :return: A dictionary mapping implementation names to their corresponding metric computation functions.
        :rtype: dict[str, Callable[..., float | torch.Tensor | np.ndarray]]

        """
        raise NotImplementedError("Subclasses should implement this method.")


class MetricList:
    """A list of metrics."""

    def __init__(self, list_name: str = "", metrics: list[Metric] | None = None) -> None:
        self.list_name = list_name
        self.metrics = metrics if metrics is not None else []

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> list[float | torch.Tensor | np.ndarray | dict[str, float | torch.Tensor | np.ndarray]]:
        if len(self.metrics) == 0:
            logger.warning(f"MetricList {self.list_name} is empty. Please add metrics before calling.")
        elif len(self.metrics) == 1:
            logger.debug(
                f"MetricList {self.list_name} contains only one metric. Consider adding multiple othrogonal metrics \
                instead of computing just one metric for more comprehensive evaluation."
            )

        return [m(*args, **kwargs) for m in self.metrics]

    def __str__(self) -> str:
        return f"MetricList(name={self.list_name}, metrics={[str(m) for m in self.metrics]})"

    def fingerprint(self) -> dict[str, Any]:
        return {
            "list_name": self.list_name,
            "metrics": [m.fingerprint() for m in self.metrics],
        }
