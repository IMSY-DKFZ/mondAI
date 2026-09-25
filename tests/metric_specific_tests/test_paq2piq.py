# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from mondAI.metrics.no_reference.paq2piq import PaQ2PiQ


@pytest.mark.parametrize(
    "model_weights_url", ["https://github.com/baidut/PaQ-2-PiQ/releases/download/v1.0/RoIPoolModel-fit.10.bs.120.pth"]
)
def test_valid_url(model_weights_url: str) -> None:
    paq2piq = PaQ2PiQ(model_weights_url=model_weights_url)
    img1 = torch.rand(3, 64, 64)
    paq2piq(img1, dims=["C", "H", "W"])


@pytest.mark.parametrize("model_weights_url", ["ftp://example.com/model_weights.pth"])
def test_invalid_url(model_weights_url: str) -> None:
    with pytest.raises(ValueError):
        PaQ2PiQ(model_weights_url=model_weights_url)


@pytest.mark.parametrize("factor", [1.1, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        paq2piq = PaQ2PiQ()
        img1 = torch.ones(3, 64, 64) * factor
        paq2piq(img1, dims=["C", "H", "W"])


def test_rgb_valid() -> None:
    paq2piq = PaQ2PiQ()
    img1 = torch.rand(3, 64, 64)
    paq2piq(img1, dims=["C", "H", "W"])
    img1 = torch.rand(64, 64, 3)
    paq2piq(img1, dims=["H", "W", "C"])


def test_rgb_invalid_dims() -> None:
    with pytest.raises(ValueError):
        paq2piq = PaQ2PiQ()
        img1 = torch.rand(64, 64)
        paq2piq(img1, dims=["H", "W"])


@pytest.mark.parametrize("n_channels", [1, 4])
def test_rgb_invalid_channels(n_channels: int) -> None:
    with pytest.raises(ValueError):
        paq2piq = PaQ2PiQ()
        img1 = torch.rand(n_channels, 64, 64)
        paq2piq(img1, dims=["C", "H", "W"])
