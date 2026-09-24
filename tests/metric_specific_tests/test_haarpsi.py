# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from mondAI.metrics.full_reference.haarpsi import HaarPSI


def test_rgb_valid() -> None:
    haarpsi = HaarPSI(use_rgb=True)
    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)
    haarpsi(img1, img2, dims=["C", "H", "W"])


def test_rgb_invalid_dims() -> None:
    with pytest.raises(ValueError):
        haarpsi = HaarPSI(use_rgb=True)
        img1 = torch.rand(64, 64)
        img2 = torch.rand(64, 64)
        haarpsi(img1, img2, dims=["H", "W"])


def test_rgb_invalid_channels() -> None:
    with pytest.raises(ValueError):
        haarpsi = HaarPSI(use_rgb=True)
        img1 = torch.rand(4, 64, 64)
        img2 = torch.rand(4, 64, 64)
        haarpsi(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("C", [1.0, 30.0, 33.3, 100.0, 30])
def test_C_valid(C: float) -> None:
    haarpsi = HaarPSI(C=C)
    img1 = torch.rand(64, 64)
    img2 = torch.rand(64, 64)
    haarpsi(img1, img2)


@pytest.mark.parametrize("C", [-1.0, 0.0])
def test_C_invalid(C: float) -> None:
    with pytest.raises(ValueError):
        HaarPSI(C=C)


@pytest.mark.parametrize("alpha", [0.1, 4.2, 10.0])
def test_alpha_valid(alpha: float) -> None:
    haarpsi = HaarPSI(alpha=alpha)
    img1 = torch.rand(64, 64)
    img2 = torch.rand(64, 64)
    haarpsi(img1, img2)


@pytest.mark.parametrize("alpha", [-0.1, 0.0, True])
def test_alpha_invalid(alpha: float) -> None:
    with pytest.raises(ValueError):
        HaarPSI(alpha=alpha)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        haarpsi = HaarPSI()
        img1 = torch.ones(64, 64) * factor
        img2 = torch.ones(64, 64)
        haarpsi(img1, img2)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        haarpsi = HaarPSI()
        img1 = torch.ones(64, 64)
        img2 = torch.ones(64, 64) * factor
        haarpsi(img1, img2)


def test_image_too_small() -> None:
    haarpsi = HaarPSI()
    img1 = torch.ones(14, 15)
    img2 = torch.ones(14, 15)
    haarpsi(img1, img2)
