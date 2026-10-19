# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from mondAI.metrics.full_reference.s3im import S3IM


@pytest.mark.parametrize("sensitivity", [0.0, 0.001, 0.05, 0.5])
def test_sensitivity_valid(sensitivity: float) -> None:
    s3im = S3IM(sensitivity=sensitivity)
    img1 = torch.rand(64, 64)
    img2 = torch.rand(64, 64)
    s3im(img1, img2)


@pytest.mark.parametrize("sensitivity", [-1, -0.001])
def test_sensitivity_invalid(sensitivity: float) -> None:
    with pytest.raises(ValueError):
        S3IM(sensitivity=sensitivity)


def test_identical_images_are_one() -> None:
    s3im = S3IM()
    img = torch.rand(64, 64)
    result = s3im(img, img)
    assert torch.isclose(torch.as_tensor(result), torch.tensor(1.0, dtype=torch.as_tensor(result).dtype))
