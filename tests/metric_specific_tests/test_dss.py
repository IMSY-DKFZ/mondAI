import pytest
import torch

from mondAI.metrics.full_reference.dss import DSS


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        dss = DSS()
        img1 = torch.ones(64, 64) * factor
        img2 = torch.ones(64, 64)
        dss(img1, img2)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        dss = DSS()
        img1 = torch.ones(64, 64)
        img2 = torch.ones(64, 64) * factor
        dss(img1, img2)


@pytest.mark.parametrize("sigma", [0.003, 0.05, 1.0, 1.55, 15.0])
def test_sigma_valid(sigma: float) -> None:
    dss = DSS(sigma=sigma)
    img1 = torch.rand(64, 64)
    img2 = torch.rand(64, 64)
    dss(img1, img2)


@pytest.mark.parametrize("sigma", [0.0, -1, -0.1])
def test_sigma_invalid(sigma: float) -> None:
    with pytest.raises(ValueError):
        DSS(sigma=sigma)


@pytest.mark.parametrize("C", [(1.0, 3.0), (1000, 300), (0.1, 0.01)])
def test_C_valid(C: tuple[float, float]) -> None:
    dss = DSS(C=C)
    img1 = torch.rand(64, 64)
    img2 = torch.rand(64, 64)
    dss(img1, img2)


@pytest.mark.parametrize("C", [(0.0, 3.0), (1000, 0.0), (-1.0, 300), (1000, -1.0)])
def test_C_invalid(C: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        DSS(C=C)
