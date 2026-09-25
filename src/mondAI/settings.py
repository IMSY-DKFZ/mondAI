# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

from mondAI.logging import update_log_level
from mondAI.metrics.dimension import _DIMENSION_LOOKUP


class _Settings:
    def __init__(self) -> None:
        # Default values for reset

        self._default_torch_device = "gpu"  # Options: "cpu", "gpu"
        self._default_logging_level = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
        self._default_tolerance = 1e-8  # Positive float
        self._default_dimensions = ("H", "W")  # Default dimensions for metrics, can be overridden in metric calls

        # Initial settings
        self._torch_device: str = self._default_torch_device
        self._logging_level: str = self._default_logging_level
        self._tolerance: float = self._default_tolerance
        self._default_dims: tuple[str, ...] = self._default_dimensions

    @property
    def torch_device(self) -> str:
        return self._torch_device

    @torch_device.setter
    def torch_device(self, device: str) -> None:
        if device not in ["cpu", "gpu"]:
            raise ValueError(f"torch_device must be 'cpu' or 'gpu', but is {device}")
        self._torch_device = device

    @property
    def logging_level(self) -> str:
        return self._logging_level

    @logging_level.setter
    def logging_level(self, level: str) -> None:
        if level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"logging_level must be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', but is {level}")
        self._logging_level = level
        update_log_level(self._logging_level)

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tol: float) -> None:
        if tol <= 0:
            raise ValueError(f"tolerance must be positive, but is {tol}")
        if not isinstance(tol, float):
            raise ValueError(f"tolerance must be a float, but is {type(tol)}")
        self._tolerance = tol

    @property
    def default_dims(self) -> tuple[str, ...]:
        return self._default_dims

    @default_dims.setter
    def default_dims(self, dims: tuple[str, ...]) -> None:
        if not all(isinstance(dim, str) for dim in dims):
            raise ValueError(f"All default dimensions must be strings, but got {dims}")
        if not all(dim in _DIMENSION_LOOKUP for dim in dims):
            raise ValueError(f"All default dimensions must be one of {list(_DIMENSION_LOOKUP.keys())}, but got {dims}")
        self._default_dims = dims

    # Reset all settings to defaults
    def reset_to_defaults(self) -> None:
        self._torch_device = self._default_torch_device
        self._logging_level = self._default_logging_level
        self._tolerance = self._default_tolerance
        self._default_dims = self._default_dimensions


settings = _Settings()
