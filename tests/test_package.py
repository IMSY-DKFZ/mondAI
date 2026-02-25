"""Test the mondAI package's __getattr__ and __setattr__ functions, as well as the
__version__ attribute."""

import pytest

from mondAI import __getattr__, __setattr__
from mondAI.settings import settings as _settings


def test_setattr_existing_attribute() -> None:
    # Arrange
    attribute_name = "torch_device"
    new_value = "cpu"
    setattr(_settings, attribute_name, new_value)

    # Act
    __setattr__(attribute_name, new_value)

    # Assert
    assert getattr(_settings, attribute_name) == new_value


def test_setattr_overwrite_existing_attribute() -> None:
    # Arrange
    attribute_name = "torch_device"
    initial_value = "cpu"
    new_value = "gpu"
    setattr(_settings, attribute_name, initial_value)  # Set initial value

    # Act
    __setattr__(attribute_name, new_value)

    # Assert
    assert getattr(_settings, attribute_name) == new_value


def test_getattr_existing_attribute() -> None:
    # Arrange
    attribute_name = "torch_device"
    expected_value = "cpu"
    setattr(_settings, attribute_name, expected_value)

    # Act
    result = __getattr__(attribute_name)

    # Assert
    assert result == expected_value


def test_getattr_non_existing_attribute() -> None:
    # Arrange
    attribute_name = "non_existing_attribute"

    # Act & Assert
    with pytest.raises(AttributeError):
        __getattr__(attribute_name)


def test_version() -> None:
    from mondAI import __version__

    assert isinstance(__version__, str)
