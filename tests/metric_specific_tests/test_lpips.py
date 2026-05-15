from typing import Literal

import pytest
import torch

from mondAI.metrics.full_reference.lpips import LPIPS


def test_rgb_valid() -> None:
    lpips = LPIPS()
    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)
    lpips(img1, img2, dims=["C", "H", "W"])


def test_rgb_valid_large() -> None:
    lpips = LPIPS()
    img1 = torch.rand(3, 400, 600)
    img2 = torch.rand(3, 400, 600)
    lpips(img1, img2, dims=["C", "H", "W"])


def test_rgb_invalid_dims() -> None:
    with pytest.raises(ValueError):
        lpips = LPIPS()
        img1 = torch.rand(64, 64)
        img2 = torch.rand(64, 64)
        lpips(img1, img2, dims=["H", "W"])


def test_rgb_invalid_channels() -> None:
    with pytest.raises(ValueError):
        lpips = LPIPS()
        img1 = torch.rand(4, 64, 64)
        img2 = torch.rand(4, 64, 64)
        lpips(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("factor", [255.0, 256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        lpips = LPIPS()
        img1 = torch.ones(3, 64, 64) * factor
        img2 = torch.ones(3, 64, 64)
        lpips(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("factor", [255.0, 256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        lpips = LPIPS()
        img1 = torch.ones(3, 64, 64)
        img2 = torch.ones(3, 64, 64) * factor
        lpips(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("network_architecture", ["alex", "vgg", "squeeze"])
def test_rgb_valid_architectures(network_architecture: Literal["alex", "vgg", "squeeze"]) -> None:
    lpips = LPIPS(network_architecture=network_architecture)
    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)
    lpips(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("network_architecture", ["alexnet", "vgg16", "unet"])
def test_rgb_invalid_architectures(network_architecture: Literal["alex", "vgg", "squeeze"]) -> None:
    with pytest.raises(ValueError):
        LPIPS(network_architecture=network_architecture)
