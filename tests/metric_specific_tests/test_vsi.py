# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from mondAI.metrics.full_reference.vsi import VSI


def test_rgb_valid() -> None:
    vsi = VSI()
    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)
    vsi(img1, img2, dims=["C", "H", "W"])


def test_rgb_invalid_dims() -> None:
    with pytest.raises(ValueError):
        vsi = VSI()
        img1 = torch.rand(64, 64)
        img2 = torch.rand(64, 64)
        vsi(img1, img2, dims=["H", "W"])


def test_rgb_invalid_channels() -> None:
    with pytest.raises(ValueError):
        vsi = VSI()
        img1 = torch.rand(4, 64, 64)
        img2 = torch.rand(4, 64, 64)
        vsi(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        vsi = VSI()
        img1 = torch.ones(3, 64, 64) * factor
        img2 = torch.ones(3, 64, 64)
        vsi(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        vsi = VSI()
        img1 = torch.ones(3, 64, 64)
        img2 = torch.ones(3, 64, 64) * factor
        vsi(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize(
    "parameter, values",
    [
        ("c1", [0.0, 1.27, 0.1]),
        ("c2", [0.0, 2.0, 610.0]),
        ("c3", [0.0, 2.0, 610.0]),
        ("alpha", [0.4, 0.02, 1.0, 0.0]),
        ("beta", [0.4, 0.02, 1.0, 0.0]),
    ],
)
def test_parameter_valid(
    parameter: str,
    values: list[float],
) -> None:
    for value in values:
        kwargs = {parameter: value}
        vsi = VSI(**kwargs)
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        vsi(img1, img2, dims=["C", "H", "W"])


@pytest.mark.parametrize(
    "parameter, values",
    [
        ("c1", [-0.1, -1.0]),
        ("c2", [-0.1, -1.0]),
        ("c3", [-0.1, -1.0]),
        ("alpha", [-0.1, -1.0]),
        ("beta", [-0.1, -1.0]),
    ],
)
def test_parameter_invalid(
    parameter: str,
    values: list[float],
) -> None:
    for value in values:
        kwargs = {parameter: value}
        with pytest.raises(ValueError):
            VSI(**kwargs)
