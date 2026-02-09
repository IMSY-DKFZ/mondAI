import numpy as np
import torch

from mondAI.settings import settings
from mondAI.utils.internal_format import convert_to_internal_format


def test_convert_to_internal_format_numpy() -> None:
    # Test with a numpy array
    image_np = np.random.rand(2, 3, 4, 5)  # Random numpy array
    result = convert_to_internal_format(image_np)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float64
    assert result.shape == (2, 3, 4, 5)


def test_convert_to_internal_format_tensor() -> None:
    # Test with a torch tensor
    image_tensor = torch.rand(2, 3, 4, 5, dtype=torch.float32)  # Random torch tensor
    result = convert_to_internal_format(image_tensor)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float64
    assert result.shape == (2, 3, 4, 5)


def test_convert_to_internal_format_device() -> None:
    # Test if the tensor is moved to the correct device
    image_np = np.random.rand(2, 3, 4, 5)
    result = convert_to_internal_format(image_np)
    assert (
        result.device.type
        == torch.device("cuda" if torch.cuda.is_available() and settings.torch_device == "gpu" else "cpu").type
    )


def test_convert_to_internal_format_settings_cpu() -> None:
    # Test with settings.torch_device set to 'cpu'
    from mondAI.settings import settings

    original_device = settings.torch_device
    settings.torch_device = "cpu"

    image_np = np.random.rand(2, 3, 4, 5)
    result = convert_to_internal_format(image_np)
    assert result.device == torch.device("cpu")

    # Restore original setting
    settings.torch_device = original_device
