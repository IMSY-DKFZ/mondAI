# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch

from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric


class MetricTemplate(FullReferenceMetric):  # TODO: rename to actual metric name
    """Template for a full reference metric implementation.

    This class serves as a template for implementing new full reference metrics. It provides the necessary structure and
    methods that need to be implemented for a new metric. To create a new metric, simply copy this class and fill in the
    appropriate details such as the metric name, abbreviation, expected dimensions, and the computation logic in the
    `_compute` method.

    Please describe the metric in detail here, including its mathematical definition, references to original
    publications and any relevant information about its use cases or properties. This will help users understand the
    metric and its applications. If your implementation builds upon or is inspired by existing work, please provide
    proper citations and references to the original sources.

    """

    @property
    def name(self) -> str:
        return "Metric Template"  # TODO: replace with actual metric name

    @property
    def abbreviation(self) -> str:
        return "MT"  # TODO: replace with actual metric abbreviation

    @property
    def higher_is_better(self) -> bool:
        return False  # TODO: set to True if higher metric values indicate better performance, False otherwise

    @property
    def scaling_factor(self) -> float:
        return 1.0

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        """TODO: replace with actual expected dimensions for the metric, e.g. (Dimension.HEIGHT, Dimension.WIDTH) for 2D
        single channel metrics, or (Dimension.HEIGHT, Dimension.WIDTH, Dimension.CHANNEL) for RGB metrics. The expected
        dimensions should be a tuple of Dimension enums that the metric is designed to work with. This will be used to
        validate the input images and ensure that the metric is applied correctly.
        """
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self, parameter1: int = 100) -> None:
        """Initialize the Metric Template.

        The constructor should include checks for any parameters that the metric uses
        to ensure they are within valid ranges or meet certain conditions. For example,
        if the metric has a parameter that must be non-negative, this method should
        check that condition and raise a ValueError if it is not met. This helps to
        prevent invalid configurations of the metric that could lead to incorrect
        results or errors during computation.

        Store all parameters as instance attributes with the same name so that they can
        be included in the fingerprint and string representation of the metric. This
        ensures that the metric's configuration is fully captured and can be reproduced
        accurately.

        :param parameter1: An example parameter for the metric, replace with actual
            parameters as needed.
        :type parameter1: int

        """
        super().__init__()
        self.parameter1 = parameter1  # TODO: replace with actual parameters for the metric

        # TODO: replace with actual checks for the metric's parameters.
        if self.parameter1 < 0:
            raise ValueError(f"parameter1 must be non-negative, but got {self.parameter1}.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the metric between image and reference.

        TODO: replace with actual computation logic for the metric. This method should implement the core logic of the
        metric, taking the input image and reference and returning the computed metric score as a torch.Tensor.

        Please only perform metric specific checks in this method. General checks for
        input images are already performed in the base class.

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :return: The computed metric score.
        :rtype: torch.Tensor

        """
        return torch.tensor(0.0)  # TODO: replace with actual computation logic for the metric

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        """Override this method in subclasses to register other implementations of the
        metric for comparison. Use the `_register_implementation` helper method to add
        implementations to the internal dictionary. These implementations will be used
        when compare_implementations is True.

        Store reference implementations in the `third_party` submodule of the metrics module,
        and import them here to register them for comparison.

        """
        # TODO: replace with actual other implementations of the metric, if available.
        # self._register_implementation(implementations, "library_name", get_library_metric(self.parameter1))

    def __str__(self) -> str:
        """Full text representation of the metric.

        TODO: replace with actual string representation of the metric. This should include the metric's name,
        abbreviation, an arrow indicating whether higher values are better, and all parameter values. The string
        representation is important for reporting and logging purposes, as it provides a clear and concise description
        of the metric being used.

        :return: A string representation of the metric including its name, abbreviation, and an arrow indicating
        whether higher values are better.
        :rtype: str

        """
        return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()} with parameter1={self.parameter1}"
