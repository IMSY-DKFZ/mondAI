import pytest
import torch

from mondAI.metrics.full_reference.msssim import MSSSIM


@pytest.mark.parametrize("k1", [0.0, 0.001, 0.01, 0.05])
def test_k1_valid(k1: float) -> None:
    metric = MSSSIM(k1=k1)
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


@pytest.mark.parametrize("k2", [0.0, 0.003, 0.03, 0.05])
def test_k2_valid(k2: float) -> None:
    metric = MSSSIM(k2=k2)
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


@pytest.mark.parametrize("parameter, value", [("k1", -1e-6), ("k2", -1e-6)])
def test_k_parameters_invalid(parameter: str, value: float) -> None:
    with pytest.raises(ValueError):
        MSSSIM(**{parameter: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("kernel_size", [3, 5, 11, 15])
def test_kernel_size_valid(kernel_size: int) -> None:
    metric = MSSSIM(kernel_size=kernel_size)
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


@pytest.mark.parametrize("kernel_size", [0, 1, 2, 10])
def test_kernel_size_invalid(kernel_size: int) -> None:
    with pytest.raises(ValueError):
        MSSSIM(kernel_size=kernel_size)


@pytest.mark.parametrize("kernel_sigma", [0.1, 1.0, 1.5, 3.0])
def test_kernel_sigma_valid(kernel_sigma: float) -> None:
    metric = MSSSIM(kernel_sigma=kernel_sigma)
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


@pytest.mark.parametrize("kernel_sigma", [0.0, -1.0])
def test_kernel_sigma_invalid(kernel_sigma: float) -> None:
    with pytest.raises(ValueError):
        MSSSIM(kernel_sigma=kernel_sigma)


@pytest.mark.parametrize("dynamic_range", [1.0, 100.0, 255.0])
def test_dynamic_range_valid(dynamic_range: float) -> None:
    metric = MSSSIM(dynamic_range=dynamic_range)
    img1 = torch.rand(256, 256) * dynamic_range
    img2 = torch.rand(256, 256) * dynamic_range
    metric(img1, img2)


@pytest.mark.parametrize("dynamic_range", [0.0, -1.0])
def test_dynamic_range_invalid(dynamic_range: float) -> None:
    with pytest.raises(ValueError):
        MSSSIM(dynamic_range=dynamic_range)


@pytest.mark.parametrize("levels", [1, 3, 5])
def test_levels_valid(levels: int) -> None:
    weights = MSSSIM.DEFAULT_WEIGHTS[:levels]
    metric = MSSSIM(levels=levels, weights=weights)
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


@pytest.mark.parametrize("levels", [0, -1])
def test_levels_invalid(levels: int) -> None:
    with pytest.raises(ValueError):
        MSSSIM(levels=levels)


def test_weights_length_invalid() -> None:
    with pytest.raises(ValueError):
        MSSSIM(levels=5, weights=(0.5, 0.5))


def test_weights_sum_invalid() -> None:
    with pytest.raises(ValueError):
        MSSSIM(levels=2, weights=(0.0, 0.0))


@pytest.mark.parametrize("method", ["product", "wtd_sum"])
def test_method_valid(method: str) -> None:
    metric = MSSSIM(levels=2, weights=(0.5, 0.5), method=method)
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


def test_method_invalid() -> None:
    with pytest.raises(ValueError):
        MSSSIM(method="invalid")


def test_image_too_small_for_window() -> None:
    metric = MSSSIM(kernel_size=11, levels=1, weights=(1.0,))
    img1 = torch.ones(10, 10)
    img2 = torch.ones(10, 10)
    with pytest.raises(ValueError):
        metric(img1, img2)


def test_image_too_small_for_levels() -> None:
    metric = MSSSIM(levels=5)
    img1 = torch.ones(64, 64)
    img2 = torch.ones(64, 64)
    with pytest.raises(ValueError):
        metric(img1, img2)


def test_identical_images_are_one() -> None:
    metric = MSSSIM()
    img = torch.rand(256, 256) * 255.0
    result = metric(img, img)
    assert torch.isclose(torch.as_tensor(result), torch.tensor(1.0, dtype=torch.as_tensor(result).dtype))
