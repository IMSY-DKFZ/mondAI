import numpy as np
import torch

from mondAI.settings import settings


def convert_to_internal_format(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Convert input image to internal torch tensor format with dtype float64,
      shape (B, C, H, W, D) and moves to GPU if available.

    :param image: Input image
    :type image: np.ndarray | torch.Tensor
    :return: Image as torch tensor
    :rtype: torch.Tensor
    """

    # Convert numpy array to float64 torch tensor if necessary
    if isinstance(image, np.ndarray):
        internal_format = torch.tensor(image, dtype=torch.float64, device=_get_torch_device())
    else:
        internal_format = image.to(dtype=torch.float64, device=_get_torch_device())

    return internal_format


def _get_torch_device() -> torch.device:
    """
    Get the appropriate torch device (GPU if available and not explicitly stated otherwise in settings, else CPU).

    :return: Torch device
    :rtype: torch.device
    """
    if settings.torch_device == "cpu":
        return torch.device("cpu")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
