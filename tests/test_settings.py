"""This file contains tests which verify the functionality of the settings module.

It tests the default values of the settings, as well as the validation logic in the
setters for each setting. The tests ensure that invalid values raise appropriate
exceptions and that valid values are accepted correctly. Additionally, it tests the
`reset_to_defaults` method to confirm that it restores all settings to their default
values.

"""

import pytest

from mondAI.logging import update_log_level
from mondAI.settings import _Settings


@pytest.fixture
def settings() -> _Settings:
    return _Settings()


def test_initial_settings(settings: _Settings) -> None:
    assert settings.torch_device == "gpu"
    assert settings.logging_level == "INFO"
    assert settings.tolerance == 1e-8


def test_torch_device_setter(settings: _Settings) -> None:
    settings.torch_device = "cpu"
    assert settings.torch_device == "cpu"

    with pytest.raises(ValueError):
        settings.torch_device = "invalid_device"


def test_logging_level_setter(settings: _Settings) -> None:
    settings.logging_level = "DEBUG"
    assert settings.logging_level == "DEBUG"

    with pytest.raises(ValueError):
        settings.logging_level = "invalid_level"


def test_tolerance_setter(settings: _Settings) -> None:
    settings.tolerance = 1e-5
    assert settings.tolerance == 1e-5

    with pytest.raises(ValueError):
        settings.tolerance = -1e-5

    with pytest.raises(TypeError):
        settings.tolerance = "string"  # type: ignore

    with pytest.raises(ValueError):
        settings.tolerance = 0

    with pytest.raises(ValueError):
        settings.tolerance = 1


def test_default_dims_setter(settings: _Settings) -> None:
    settings.default_dims = ("B", "C", "H", "W")
    assert settings.default_dims == ("B", "C", "H", "W")

    with pytest.raises(ValueError):
        settings.default_dims = ("B", "C", None, "W")  # type: ignore

    with pytest.raises(ValueError):
        settings.default_dims = ("B", "C", "X", "W")  # Invalid dimension


def test_reset_to_defaults(settings: _Settings) -> None:
    settings.torch_device = "cpu"
    settings.logging_level = "DEBUG"
    settings.tolerance = 1e-5
    settings.default_dims = ("B", "C", "H", "W")

    settings.reset_to_defaults()

    assert settings.torch_device == "gpu"
    assert settings.logging_level == "INFO"
    assert settings.tolerance == 1e-8
    assert settings.default_dims == ("H", "W")  # type: ignore[comparison-overlap]


def test_update_log_level_invalid_input(settings: _Settings) -> None:
    # extra test to ensure that update_log_level raises ValueError for invalid input,
    # although it's also checked by the logging_level setter
    with pytest.raises(ValueError):
        update_log_level("invalid_level")
