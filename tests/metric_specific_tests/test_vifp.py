import pytest
import torch

from mondAI.metrics.full_reference.vifp import VIFP


@pytest.mark.parametrize("sigma_n_squared", [0.0, 1.0, 30.0, 33.3, 100.0, 30])
def test_sigma_n_squared_valid(sigma_n_squared: float) -> None:
    vifp = VIFP(sigma_n_squared=sigma_n_squared)
    img1 = torch.rand(64, 64)
    img2 = torch.rand(64, 64)
    vifp(img1, img2)


@pytest.mark.parametrize("sigma_n_squared", [-1.0])
def test_sigma_n_squared_invalid(sigma_n_squared: float) -> None:
    with pytest.raises(ValueError):
        VIFP(sigma_n_squared=sigma_n_squared)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        vifp = VIFP()
        img1 = torch.ones(64, 64) * factor
        img2 = torch.ones(64, 64)
        vifp(img1, img2)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        vifp = VIFP()
        img1 = torch.ones(64, 64)
        img2 = torch.ones(64, 64) * factor
        vifp(img1, img2)


def test_image_too_small() -> None:
    vifp = VIFP()
    img1 = torch.ones(14, 15)
    img2 = torch.ones(14, 15)
    with pytest.raises(ValueError):
        vifp(img1, img2)
