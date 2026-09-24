# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import torch


def rgb_to_yiq(image: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image to YIQ color space.

    :param image: Input RGB image as a torch.Tensor with shape (C, H, W, ...) or (H, W, ..., C)
        where C=3.
    :type image: torch.Tensor
    :return: Image converted to YIQ color space with the same shape as the input.
    :rtype: torch.Tensor
    :raises ValueError: If the input image does not have 3 channels in either (C, H, W)
        or (H, W, C) format.

    """
    if image.shape[0] == 3:  # (C, H, W)
        r = image[0]
        g = image[1]
        b = image[2]
    elif image.shape[-1] == 3:  # (H, W, C)
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
    else:
        raise ValueError(
            "Input image must have 3 channels in either (C, H, W) or (H, W, C) format, "
            f"but got shape {tuple(image.shape)}."
        )

    # Conversion values are taken from https://github.com/rgcda/haarpsi/blob/master/HaarPSI.m
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.274 * g - 0.322 * b
    q = 0.211 * r - 0.523 * g + 0.312 * b

    if image.shape[0] == 3:  # (C, H, W)
        return torch.stack((y, i, q), dim=0)
    else:  # (H, W, C)
        return torch.stack((y, i, q), dim=-1)
