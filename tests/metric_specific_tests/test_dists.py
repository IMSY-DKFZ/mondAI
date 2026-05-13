import pytest
import torch

from mondAI.metrics.full_reference.dists import DISTS


def test_rgb_valid() -> None:
    dists = DISTS()
    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)
    dists(img1, img2, dims=["C", "H", "W"])


def test_rgb_valid_large() -> None:
    dists = DISTS()
    img1 = torch.rand(3, 400, 600)
    img2 = torch.rand(3, 400, 600)
    dists(img1, img2, dims=["C", "H", "W"])


def test_rgb_invalid_dims() -> None:
    with pytest.raises(ValueError):
        dists = DISTS()
        img1 = torch.rand(64, 64)
        img2 = torch.rand(64, 64)
        dists(img1, img2, dims=["H", "W"])


def test_rgb_invalid_channels() -> None:
    with pytest.raises(ValueError):
        dists = DISTS()
        img1 = torch.rand(4, 64, 64)
        img2 = torch.rand(4, 64, 64)
        dists(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("factor", [255.0, 256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        dists = DISTS()
        img1 = torch.ones(3, 64, 64) * factor
        img2 = torch.ones(3, 64, 64)
        dists(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("factor", [255.0, 256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        dists = DISTS()
        img1 = torch.ones(3, 64, 64)
        img2 = torch.ones(3, 64, 64) * factor
        dists(img1, img2, dims=["C", "H", "W"])
