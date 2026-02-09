import pytest

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


def test_reset_to_defaults(settings: _Settings) -> None:
    settings.torch_device = "cpu"
    settings.logging_level = "DEBUG"
    settings.tolerance = 1e-5

    settings.reset_to_defaults()

    assert settings.torch_device == "gpu"
    assert settings.logging_level == "INFO"
    assert settings.tolerance == 1e-8
