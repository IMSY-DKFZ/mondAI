import pytest
import torch

from mondAI.metrics.full_reference.iwssim import IWSSIM
from mondAI.metrics.full_reference.ssim import SSIM


@pytest.mark.parametrize("k1", [0.0, 0.001, 0.01, 0.05])
def test_k1_valid(k1: float) -> None:
    metric = IWSSIM(k1=k1)
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


@pytest.mark.parametrize("k2", [0.0, 0.003, 0.03, 0.05])
def test_k2_valid(k2: float) -> None:
    metric = IWSSIM(k2=k2)
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


@pytest.mark.parametrize("parameter, value", [("k1", -1e-6), ("k2", -1e-6)])
def test_k_parameters_invalid(parameter: str, value: float) -> None:
    with pytest.raises(ValueError):
        IWSSIM(**{parameter: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("kernel_size", [3, 5, 11, 15])
def test_kernel_size_valid(kernel_size: int) -> None:
    metric = IWSSIM(kernel_size=kernel_size, scales=3, weights=IWSSIM.DEFAULT_WEIGHTS[:3])
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


@pytest.mark.parametrize("kernel_size", [0, 1, 2, 10])
def test_kernel_size_invalid(kernel_size: int) -> None:
    with pytest.raises(ValueError):
        IWSSIM(kernel_size=kernel_size)


@pytest.mark.parametrize("kernel_sigma", [-1, 0.0])
def test_kernel_sigma_invalid(kernel_sigma: float) -> None:
    with pytest.raises(ValueError):
        IWSSIM(kernel_sigma=kernel_sigma)


@pytest.mark.parametrize("dynamic_range", [1.0, 100.0, 255.0])
def test_dynamic_range_valid(dynamic_range: float) -> None:
    metric = IWSSIM(dynamic_range=dynamic_range)
    img1 = torch.rand(256, 256) * dynamic_range
    img2 = torch.rand(256, 256) * dynamic_range
    metric(img1, img2)


@pytest.mark.parametrize("dynamic_range", [0.0, -1.0])
def test_dynamic_range_invalid(dynamic_range: float) -> None:
    with pytest.raises(ValueError):
        IWSSIM(dynamic_range=dynamic_range)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        ssim = IWSSIM()
        img1 = torch.ones(64, 64) * factor
        img2 = torch.ones(64, 64)
        ssim(img1, img2)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        ssim = IWSSIM()
        img1 = torch.ones(64, 64)
        img2 = torch.ones(64, 64) * factor
        ssim(img1, img2)


@pytest.mark.parametrize("scales", [1, 3, 5])
def test_scales_valid(scales: int) -> None:
    metric = IWSSIM(scales=scales, weights=IWSSIM.DEFAULT_WEIGHTS[:scales])
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


@pytest.mark.parametrize("scales", [0, -1])
def test_scales_invalid(scales: int) -> None:
    with pytest.raises(ValueError):
        IWSSIM(scales=scales)


def test_weights_length_invalid() -> None:
    with pytest.raises(ValueError):
        IWSSIM(scales=5, weights=(0.5, 0.5))


def test_weights_sum_invalid() -> None:
    with pytest.raises(ValueError):
        IWSSIM(scales=2, weights=(0.0, 0.0))


@pytest.mark.parametrize("sigma_n_squared", [0.0, 0.4, 1.0])
def test_sigma_n_squared_valid(sigma_n_squared: float) -> None:
    metric = IWSSIM(sigma_n_squared=sigma_n_squared)
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    metric(img1, img2)


def test_sigma_n_squared_invalid() -> None:
    with pytest.raises(ValueError):
        IWSSIM(sigma_n_squared=-1.0)


@pytest.mark.parametrize("block_size", [0, 2, 4, -1])
def test_block_size_invalid(block_size: int) -> None:
    with pytest.raises(ValueError):
        IWSSIM(block_size=block_size)


def test_image_too_small_for_window() -> None:
    metric = IWSSIM(kernel_size=11, scales=1, weights=(1.0,))
    img1 = torch.ones(10, 10)
    img2 = torch.ones(10, 10)
    with pytest.raises(ValueError):
        metric(img1, img2)


def test_image_too_small_for_scales() -> None:
    metric = IWSSIM(scales=5)
    img1 = torch.ones(64, 64)
    img2 = torch.ones(64, 64)
    with pytest.raises(ValueError):
        metric(img1, img2)


def test_identical_images_are_one() -> None:
    metric = IWSSIM(scales=3, weights=IWSSIM.DEFAULT_WEIGHTS[:3])
    img = torch.rand(256, 256) * 255.0
    result = metric(img, img)
    assert torch.isclose(torch.as_tensor(result), torch.tensor(1.0, dtype=torch.as_tensor(result).dtype), atol=1e-6)


def test_single_scale_same_as_ssim() -> None:
    iwssim = IWSSIM(scales=1, weights=(1.0,))
    ssim = SSIM()
    img1 = torch.rand(256, 256) * 255.0
    img2 = torch.rand(256, 256) * 255.0
    score_iwssim = iwssim(img1, img2)
    score_ssim = ssim(img1, img2)
    assert torch.isclose(torch.as_tensor(score_iwssim), torch.as_tensor(score_ssim))
