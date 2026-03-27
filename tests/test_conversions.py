"""Tests the `check_*` functions in `utils/checks.py`."""

import re

import pytest
import torch

from mondAI.utils.conversions import rgb_to_yiq

### rgb_to_yiq tests ###


@pytest.mark.parametrize("shape", [(3, 64, 64), (64, 64, 3), (10, 64, 64, 3)])
def test_rgb_to_yiq_valid(shape: tuple[int, ...]) -> None:
    image_rgb = torch.rand(shape)
    image_yiq = rgb_to_yiq(image_rgb)
    assert image_rgb.shape == image_yiq.shape
    assert image_rgb.dtype == image_yiq.dtype
    assert image_rgb.device == image_yiq.device
    assert type(image_rgb) is type(image_yiq)


@pytest.mark.parametrize("shape", [(64, 64)])
def test_rgb_to_yiq_invalid(shape: tuple[int, ...]) -> None:
    image_rgb = torch.rand(shape)
    with pytest.raises(
        ValueError, match=re.escape("Input image must have 3 channels in either (C, H, W) or (H, W, C) format.")
    ):
        rgb_to_yiq(image_rgb)
