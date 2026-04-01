import pytest
import torch

from mondAI.metrics.full_reference.ssim import SSIM


@pytest.mark.parametrize("k1", [0.0, 0.001, 0.01, 0.05])
def test_k1_valid(k1: float) -> None:
    ssim = SSIM(k1=k1)
    img1 = torch.rand(64, 64) * 255.0
    img2 = torch.rand(64, 64) * 255.0
    ssim(img1, img2)


@pytest.mark.parametrize("k2", [0.0, 0.003, 0.03, 0.05])
def test_k2_valid(k2: float) -> None:
    ssim = SSIM(k2=k2)
    img1 = torch.rand(64, 64) * 255.0
    img2 = torch.rand(64, 64) * 255.0
    ssim(img1, img2)


@pytest.mark.parametrize("parameter, value", [("k1", -1e-6), ("k2", -1e-6)])
def test_k_parameters_invalid(parameter: str, value: float) -> None:
    with pytest.raises(ValueError):
        SSIM(**{parameter: value})


@pytest.mark.parametrize("kernel_size", [3, 5, 11, 15])
def test_kernel_size_valid(kernel_size: int) -> None:
    ssim = SSIM(kernel_size=kernel_size)
    img1 = torch.rand(64, 64) * 255.0
    img2 = torch.rand(64, 64) * 255.0
    ssim(img1, img2)


@pytest.mark.parametrize("kernel_size", [0, 1, 2, 10])
def test_kernel_size_invalid(kernel_size: int) -> None:
    with pytest.raises(ValueError):
        SSIM(kernel_size=kernel_size)


@pytest.mark.parametrize("kernel_sigma", [0.1, 1.0, 1.5, 3.0])
def test_kernel_sigma_valid(kernel_sigma: float) -> None:
    ssim = SSIM(kernel_sigma=kernel_sigma)
    img1 = torch.rand(64, 64) * 255.0
    img2 = torch.rand(64, 64) * 255.0
    ssim(img1, img2)


@pytest.mark.parametrize("kernel_sigma", [0.0, -1.0])
def test_kernel_sigma_invalid(kernel_sigma: float) -> None:
    with pytest.raises(ValueError):
        SSIM(kernel_sigma=kernel_sigma)


@pytest.mark.parametrize("dynamic_range", [1.0, 100.0, 255.0])
def test_dynamic_range_valid(dynamic_range: float) -> None:
    ssim = SSIM(dynamic_range=dynamic_range)
    img1 = torch.rand(64, 64) * dynamic_range
    img2 = torch.rand(64, 64) * dynamic_range
    ssim(img1, img2)


@pytest.mark.parametrize("dynamic_range", [0.0, -1.0])
def test_dynamic_range_invalid(dynamic_range: float) -> None:
    with pytest.raises(ValueError):
        SSIM(dynamic_range=dynamic_range)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        ssim = SSIM()
        img1 = torch.ones(64, 64) * factor
        img2 = torch.ones(64, 64)
        ssim(img1, img2)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        ssim = SSIM()
        img1 = torch.ones(64, 64)
        img2 = torch.ones(64, 64) * factor
        ssim(img1, img2)


def test_image_too_small() -> None:
    ssim = SSIM(kernel_size=11)
    img1 = torch.ones(10, 10)
    img2 = torch.ones(10, 10)
    with pytest.raises(ValueError):
        ssim(img1, img2)


def test_identical_images_are_one() -> None:
    ssim = SSIM()
    img = torch.rand(64, 64) * 255.0
    result = ssim(img, img)
    assert torch.isclose(torch.as_tensor(result), torch.tensor(1.0, dtype=torch.as_tensor(result).dtype))
