import pytest
import torch

from mondAI.metrics.full_reference.fsim import FSIM


def test_rgb_valid() -> None:
    fsim = FSIM(use_rgb=True)
    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)
    fsim(img1, img2, dims=["C", "H", "W"])


def test_rgb_invalid_dims() -> None:
    with pytest.raises(ValueError):
        fsim = FSIM(use_rgb=True)
        img1 = torch.rand(64, 64)
        img2 = torch.rand(64, 64)
        fsim(img1, img2, dims=["H", "W"])


def test_rgb_invalid_channels() -> None:
    with pytest.raises(ValueError):
        fsim = FSIM(use_rgb=True)
        img1 = torch.rand(4, 64, 64)
        img2 = torch.rand(4, 64, 64)
        fsim(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("T", [0.85, 0.5, 0.01, 1.0])
def test_Ts_valid(T: float) -> None:
    for t in range(4):
        fsim = FSIM(**{f"T{t + 1}": T})  # type: ignore[arg-type]
        img1 = torch.rand(64, 64)
        img2 = torch.rand(64, 64)
        fsim(img1, img2)


@pytest.mark.parametrize("T", [0.0, -0.1])
def test_Ts_invalid(T: float) -> None:
    for t in range(4):
        with pytest.raises(ValueError):
            FSIM(**{f"T{t + 1}": T})  # type: ignore[arg-type]


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        fsim = FSIM()
        img1 = torch.ones(64, 64) * factor
        img2 = torch.ones(64, 64)
        fsim(img1, img2)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        fsim = FSIM()
        img1 = torch.ones(64, 64)
        img2 = torch.ones(64, 64) * factor
        fsim(img1, img2)


@pytest.mark.parametrize(
    "parameter, values",
    [
        ("_lambda", [0.0, 0.1, 0.5, 1.0]),
        ("scales", [1, 2, 6, 10]),
        ("orientations", [1, 2, 6, 10]),
        ("minimal_wavelength", [1, 2, 6, 10]),
        ("filter_scaling_factor", [2, 6, 10]),
        ("sigma_f", [0.1, 0.5, 1.0, 2.0]),
        ("delta_theta", [0.1, 0.5, 1.0, 2.0]),
        ("noise_threshold_factor", [0.1, 0.5, 1.0]),
        ("epsilon", [1e-10, 1e-8, 1e-6, 1e-4]),
    ],
)
def test_parameter_valid(
    parameter: str,
    values: list[int | float],
) -> None:
    for value in values:
        kwargs = {parameter: value}
        fsim = FSIM(**kwargs)  # type: ignore[arg-type]
        img1 = torch.rand(64, 64)
        img2 = torch.rand(64, 64)
        fsim(img1, img2)


@pytest.mark.parametrize(
    "parameter, values",
    [
        ("_lambda", [-0.1, -1]),
        ("scales", [0, -1]),
        ("orientations", [0, -1]),
        ("minimal_wavelength", [0, -1]),
        ("filter_scaling_factor", [1, 0, -1]),
        ("sigma_f", [0.0, -0.1]),
        ("delta_theta", [0.0, -0.1]),
        ("noise_threshold_factor", [0.0, -0.1]),
        ("epsilon", [0.1, 0.5, 1.0]),
    ],
)
def test_parameter_invalid(
    parameter: str,
    values: list[int | float],
) -> None:
    for value in values:
        kwargs = {parameter: value}
        with pytest.raises(ValueError):
            FSIM(**kwargs)  # type: ignore[arg-type]
