# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from mondAI.utils.signal_processing import convolve2d, gaussian_filter_kernel


def test_gaussian_filter_kernel_sum_is_one() -> None:
    kernel = gaussian_filter_kernel(kernel_size=5, sigma=1.0, device="cpu", dtype=torch.float64)
    assert torch.sum(kernel).item() == pytest.approx(1.0, abs=1e-6)


def test_gaussian_filter_kernel_symmetry() -> None:
    kernel = gaussian_filter_kernel(kernel_size=5, sigma=1.0, device="cpu", dtype=torch.float64)
    assert torch.allclose(kernel, torch.flip(kernel, dims=[0, 1]), atol=1e-6)


@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_gaussian_filter_kernel_valid_sizes(size: int) -> None:
    kernel = gaussian_filter_kernel(kernel_size=size, sigma=1.0, device="cpu", dtype=torch.float64)
    assert kernel.shape == (size, size)


def test_convolve2d_identity() -> None:
    image = torch.rand(64, 64)
    kernel = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    result = convolve2d(image, kernel, padding="same")
    assert torch.allclose(result, image, atol=1e-6)


def test_convolve_shapes_padding_valid() -> None:
    image = torch.rand(64, 64)
    kernel = torch.rand(5, 5)
    result = convolve2d(image, kernel, padding="valid")
    assert result.shape == (60, 60)  # valid convolution reduces size by kernel_size - 1


def test_convolve_shapes_padding_same() -> None:
    image = torch.rand(64, 64)
    kernel = torch.rand(5, 5)
    result = convolve2d(image, kernel, padding="same")
    assert result.shape == (64, 64)  # same convolution preserves size
