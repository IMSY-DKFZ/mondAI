import torch


def convolve2d(image: torch.Tensor, kernel: torch.Tensor, padding: str = "valid") -> torch.Tensor:
    """Convolve the input image with the given kernel using 2D convolution.

    :param image: The input 2D image to be convolved, shape (H, W)
    :type image: torch.Tensor
    :param kernel: The 2D convolution kernel
    :type kernel: torch.Tensor
    :param padding: The padding mode for the convolution, defaults to "valid". Can be
        "valid" (no padding) or "same" (pad to keep same output size as input).
    :type padding: str
    :return: The convolved image, shape (H, W)
    :rtype: torch.Tensor

    """
    convolved = torch.nn.functional.conv2d(image.unsqueeze(0), weight=kernel.unsqueeze(0).unsqueeze(0), padding=padding)
    return convolved.squeeze()


def gaussian_filter_kernel(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate a 2D Gaussian filter kernel.

    :param kernel_size: The size of the kernel (must be odd).
    :type kernel_size: int
    :param sigma: The standard deviation of the Gaussian distribution.
    :type sigma: float
    :param device: The device on which to create the kernel.
    :type device: torch.device
    :param dtype: The data type of the kernel.
    :type dtype: torch.dtype
    :return: A 2D Gaussian filter kernel.
    :rtype: torch.Tensor

    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / torch.sum(kernel)
