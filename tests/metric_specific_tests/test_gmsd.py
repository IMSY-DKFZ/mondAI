import pytest
import torch

from mondAI.metrics.full_reference.gmsd import GMSD


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        gmsd = GMSD()
        img1 = torch.ones(64, 64) * factor
        img2 = torch.ones(64, 64)
        gmsd(img1, img2)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        gmsd = GMSD()
        img1 = torch.ones(64, 64)
        img2 = torch.ones(64, 64) * factor
        gmsd(img1, img2)


@pytest.mark.parametrize("t", [170.0, 3.0, 0.0])
def test_t_valid(t: float) -> None:
    gmsd = GMSD(t=t)
    img1 = torch.rand(64, 64) * 255.0
    img2 = torch.rand(64, 64) * 255.0
    gmsd(img1, img2)


@pytest.mark.parametrize("t", [-1.0, -0.01])
def test_t_invalid(t: float) -> None:
    with pytest.raises(ValueError):
        GMSD(t=t)
