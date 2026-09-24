# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from mondAI.metrics.full_reference.psnr import PSNR


@pytest.mark.parametrize("dynamic_range", [1.0, 100.0, 255.0])
def test_dynamic_range_valid(dynamic_range: float) -> None:
    psnr = PSNR(dynamic_range=dynamic_range)
    img1 = torch.rand(64, 64) * dynamic_range / 255.0  # compensate for internal scaling factor
    img2 = torch.rand(64, 64) * dynamic_range / 255.0  # compensate for internal scaling factor
    psnr(img1, img2)


@pytest.mark.parametrize("dynamic_range", [0.0, -1.0])
def test_dynamic_range_invalid(dynamic_range: float) -> None:
    with pytest.raises(ValueError):
        PSNR(dynamic_range=dynamic_range)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        psnr = PSNR()
        img1 = torch.ones(64, 64) * factor
        img2 = torch.ones(64, 64)
        psnr(img1, img2)


@pytest.mark.parametrize("factor", [256.0, -1.0])
def test_invalid_value_range_reference(factor: float) -> None:
    with pytest.raises(ValueError):
        psnr = PSNR()
        img1 = torch.ones(64, 64)
        img2 = torch.ones(64, 64) * factor
        psnr(img1, img2)


def test_identical_images_are_infinite() -> None:
    psnr = PSNR()
    img = torch.rand(64, 64)
    result = psnr(img, img)
    assert torch.isinf(torch.as_tensor(result))


def test_psnr_decreases_with_larger_error() -> None:
    psnr = PSNR(dynamic_range=255.0)
    reference = torch.zeros(32, 32)
    image_small_error = torch.full((32, 32), 0.1)
    image_large_error = torch.full((32, 32), 0.2)

    result_small_error = torch.as_tensor(psnr(image_small_error, reference))
    result_large_error = torch.as_tensor(psnr(image_large_error, reference))

    assert result_small_error > result_large_error
