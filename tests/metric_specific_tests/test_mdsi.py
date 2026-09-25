# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from mondAI.metrics.full_reference.mdsi import MDSI


def test_rgb_valid() -> None:
    mdsi = MDSI()
    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)
    mdsi(img1, img2, dims=["C", "H", "W"])


def test_rgb_invalid_dims() -> None:
    with pytest.raises(ValueError):
        mdsi = MDSI()
        img1 = torch.rand(64, 64)
        img2 = torch.rand(64, 64)
        mdsi(img1, img2, dims=["H", "W"])


def test_rgb_invalid_channels() -> None:
    with pytest.raises(ValueError):
        mdsi = MDSI()
        img1 = torch.rand(4, 64, 64)
        img2 = torch.rand(4, 64, 64)
        mdsi(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("combination_method", ["product", "sum"])
def test_combination_method_valid(combination_method: str) -> None:
    mdsi = MDSI(combination_method=combination_method)
    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)
    mdsi(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("combination_method", ["prod", "mult", "avg", "add"])
def test_combination_method_invalid(combination_method: str) -> None:
    with pytest.raises(ValueError):
        MDSI(combination_method=combination_method)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        mdsi = MDSI()
        img1 = torch.ones(3, 64, 64) * factor
        img2 = torch.ones(3, 64, 64)
        mdsi(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        mdsi = MDSI()
        img1 = torch.ones(3, 64, 64)
        img2 = torch.ones(3, 64, 64) * factor
        mdsi(img1, img2, dims=["C", "H", "W"])
