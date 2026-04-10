import pytest
import torch

from mondAI.metrics.full_reference.msssim import MSSSIM
from mondAI.metrics.full_reference.ssim import SSIM


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


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        ssim = MSSSIM()
        img1 = torch.ones(64, 64) * factor
        img2 = torch.ones(64, 64)
        ssim(img1, img2)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        ssim = MSSSIM()
        img1 = torch.ones(64, 64)
        img2 = torch.ones(64, 64) * factor
        ssim(img1, img2)


@pytest.mark.parametrize("scales", [1, 3, 5])
def test_scales_valid(scales: int) -> None:
    weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)[:scales]
    metric = MSSSIM(scales=scales, weights=weights)
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


@pytest.mark.parametrize("scales", [0, -1])
def test_scales_invalid(scales: int) -> None:
    with pytest.raises(ValueError):
        MSSSIM(scales=scales)


def test_weights_length_invalid() -> None:
    with pytest.raises(ValueError):
        MSSSIM(scales=5, weights=(0.5, 0.5))


def test_weights_sum_invalid() -> None:
    with pytest.raises(ValueError):
        MSSSIM(scales=2, weights=(0.0, 0.0))


@pytest.mark.parametrize("method", ["product", "weighted sum"])
def test_method_valid(method: str) -> None:
    metric = MSSSIM(scales=2, weights=(0.5, 0.5), method=method)
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


def test_method_invalid() -> None:
    with pytest.raises(ValueError):
        MSSSIM(method="invalid")


def test_image_too_small_for_window() -> None:
    metric = MSSSIM(kernel_size=11, scales=1, weights=(1.0,))
    img1 = torch.ones(10, 10)
    img2 = torch.ones(10, 10)
    with pytest.raises(ValueError):
        metric(img1, img2)


def test_image_too_small_for_scales() -> None:
    metric = MSSSIM(scales=5)
    img1 = torch.ones(64, 64)
    img2 = torch.ones(64, 64)
    with pytest.raises(ValueError):
        metric(img1, img2)


def test_identical_images_are_one() -> None:
    metric = MSSSIM()
    img = torch.rand(256, 256) * 255.0
    result = metric(img, img)
    assert torch.isclose(torch.as_tensor(result), torch.tensor(1.0, dtype=torch.as_tensor(result).dtype))


def test_single_scale_same_as_ssim() -> None:
    msssim = MSSSIM(scales=1, weights=(1.0,))
    ssim = SSIM()
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    score_msssim = msssim(img1, img2)
    score_ssim = ssim(img1, img2)
    assert torch.isclose(torch.as_tensor(score_msssim), torch.tensor(score_ssim))
