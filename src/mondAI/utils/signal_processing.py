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
    if padding == "same":
        kernel_height, kernel_width = kernel.shape
        padded_image = torch.nn.functional.pad(
            image.unsqueeze(0),
            (
                (kernel_width - 1) // 2,  # left
                (kernel_width - 1) - (kernel_width - 1) // 2,  # right
                (kernel_height - 1) // 2,  # top
                (kernel_height - 1) - (kernel_height - 1) // 2,  # bottom
            ),
        )
        convolved = torch.nn.functional.conv2d(padded_image, weight=kernel.unsqueeze(0).unsqueeze(0))
    else:
        convolved = torch.nn.functional.conv2d(
            image.unsqueeze(0), weight=kernel.unsqueeze(0).unsqueeze(0), padding=padding
        )
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
    ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / torch.sum(kernel)


def subsample(image: torch.Tensor, kernel_size: int = 2, channels: int = 1) -> torch.Tensor:
    """Subsample the input image by a factor of kernel_size (default 2) using a mean
    filter and dyadic subsampling.

    :param image: The input 2D image to be subsampled, shape (H, W),
    :type image: torch.Tensor
    :param kernel_size: The size of the mean filter kernel, defaults to 2.
    :type kernel_size: int
    :param channels: The number of channels in the input image, defaults to 1. This is
        used to create the appropriate filter weights for convolution.
    :type channels: int
    :return: The subsampled image, shape (H/k, W/k), or (H/k+1, W/k+1) if the input
        dimensions are odd.
    :rtype: torch.Tensor

    """

    filter_weights = image.new_ones(channels, 1, kernel_size, kernel_size) / kernel_size**2
    image = torch.nn.functional.pad(
        image.unsqueeze(0),
        (
            (kernel_size - 1) // 2,  # left
            (kernel_size - 1) - (kernel_size - 1) // 2,  # right
            (kernel_size - 1) // 2,  # top
            (kernel_size - 1) - (kernel_size - 1) // 2,  # bottom
        ),
    )
    mean_filtered = torch.nn.functional.conv2d(image, weight=filter_weights, groups=channels)
    return mean_filtered.squeeze()[..., ::kernel_size, ::kernel_size]  # subsampled


def gradient_map(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Compute the gradient map of the input image using the provided kernel by
    convolving the image with the kernel in both x and y directions and combining the
    results.

    :param image: The input 2D image for which to compute the gradient map, shape (H,
        W).
    :type image: torch.Tensor
    :param kernel: The 2D convolution kernel to compute the gradients, shape (k, k).
    :type kernel: torch.Tensor
    :return: The computed gradient map, shape (H, W).
    :rtype: torch.Tensor

    """
    gradient_x = convolve2d(image, kernel, padding="same")
    gradient_y = convolve2d(image, kernel.t(), padding="same")
    return torch.sqrt(gradient_x**2 + gradient_y**2)
