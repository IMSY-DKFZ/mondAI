# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from mondAI.logging import get_logger
from mondAI.settings import settings

logger = get_logger()


def convert_to_internal_format(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert input image to internal torch tensor format with dtype float64, shape
    (B, C, H, W, D) and moves to GPU if available.

    :param image: Input image
    :type image: np.ndarray | torch.Tensor
    :return: Image as torch tensor
    :rtype: torch.Tensor

    """
    # Convert numpy array to float64 torch tensor if necessary
    if isinstance(image, np.ndarray):
        logger.info(f"Converting input image from numpy array with dtype {image.dtype} to torch tensor dtype float64.")
        internal_format = torch.tensor(image, dtype=torch.float64, device=_get_torch_device())
    else:
        logger.info(
            f"Converting input image from torch tensor with dtype {image.dtype} to internal format with dtype float64."
        )
        internal_format = image.to(dtype=torch.float64, device=_get_torch_device())

    return internal_format


def _get_torch_device() -> torch.device:
    """Get the appropriate torch device (GPU if available and not explicitly stated
    otherwise in settings, else CPU).

    :return: Torch device
    :rtype: torch.device

    """
    if settings.torch_device == "cpu":
        logger.info("Using CPU as torch device as specified in settings.")
        return torch.device("cpu")
    else:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using torch device: {torch_device}.")
        return torch_device
