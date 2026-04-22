import torch

from mondAI.utils.signal_processing import convolve2d, gaussian_filter_kernel


def similarity_map(x: torch.Tensor, y: torch.Tensor, constant: float) -> torch.Tensor:
    r"""Compute the similarity map between two images x and y using this formula.

    .. math:: S(x, y, C) = \frac{2xy + C}{x^2 + y^2 + C}

    :param x: The first image, shape
    :type x: torch.Tensor
    :param y: The second image, shape
    :type y: torch.Tensor
    :param constant: A constant to stabilize the division
    :type constant: float
    :return: The similarity map between x and y, shape as x and y
    :rtype: torch.Tensor

    """

    numerator = 2.0 * x * y + constant
    denominator = x**2 + y**2 + constant
    similarity = numerator / denominator
    return similarity


def ssim_and_cs_maps(
    image: torch.Tensor,
    reference: torch.Tensor,
    *,
    kernel_size: int,
    kernel_sigma: float,
    dynamic_range: float,
    k1: float,
    k2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SSIM and contrast-structure maps for grayscale 2D images using a
    Gaussian kernel with given properties.

    Includes a fallback mechanism for cases where the stability constants lead to zero denominators.

    In cases where the constants are positive it follows these formulas:

    .. math:: \operatorname{CS}(x, y) = \frac{2\sigma_{xy} + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}
    .. math:: \operatorname{L}(x, y) = \frac{2\mu_x \mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}
    .. math:: \operatorname{SSIM}(x, y) = \operatorname{L}(x, y) \cdot \operatorname{CS}(x, y)

    where :math:`\mu_x` and :math:`\mu_y` are local means, :math:`\sigma_x^2` and
    :math:`\sigma_y^2` are local variances, and :math:`\sigma_{xy}` is the local
    covariance, :math:`CS` the contrast-structure map and :math::`L` the luminance map.
    The constants are

    .. math::
        C_1 = (K_1 L)^2, \qquad C_2 = (K_2 L)^2.

    where L is the dynamic range of the pixel values.

    :param image: The image, shape (H, W)
    :type image: torch.Tensor
    :param reference: The reference image, shape (H, W)
    :type reference: torch.Tensor
    :param kernel_size: Size of the Gaussian window, default is 11.
    :type kernel_size: int
    :param kernel_sigma: Standard deviation of the Gaussian window, default is 1.5.
    :type kernel_sigma: float
    :param dynamic_range: Dynamic range ``L`` of the images, default is 255.0.
    :type dynamic_range: float
    :param k1: First stability constant coefficient, default is 0.01.
    :type k1: float
    :param k2: Second stability constant coefficient, default is 0.03.
    :type k2: float
    :return: A tuple containing the SSIM map and the contrast-structure map, both of shape (H, W)

    """

    kernel = gaussian_filter_kernel(kernel_size, kernel_sigma, device=image.device, dtype=image.dtype)

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    mu_image = convolve2d(image, kernel, padding="valid")
    mu_reference = convolve2d(reference, kernel, padding="valid")

    mu_image_squared = mu_image * mu_image
    mu_reference_squared = mu_reference * mu_reference
    mu_image_reference = mu_image * mu_reference

    sigma_image_squared = convolve2d(image * image, kernel, padding="valid") - mu_image_squared
    sigma_reference_squared = convolve2d(reference * reference, kernel, padding="valid") - mu_reference_squared
    sigma_image_reference = convolve2d(image * reference, kernel, padding="valid") - mu_image_reference

    if c1 > 0 and c2 > 0:
        contrast_structure_map = (2 * sigma_image_reference + c2) / (sigma_image_squared + sigma_reference_squared + c2)
        luminance_map = (2 * mu_image_reference + c1) / (mu_image_squared + mu_reference_squared + c1)
        ssim_map = luminance_map * contrast_structure_map
    else:
        numerator1 = 2 * mu_image_reference + c1
        numerator2 = 2 * sigma_image_reference + c2
        denominator1 = mu_image_squared + mu_reference_squared + c1
        denominator2 = sigma_image_squared + sigma_reference_squared + c2

        contrast_structure_map = torch.ones_like(mu_image)
        valid_cs = denominator2 > 0
        contrast_structure_map[valid_cs] = numerator2[valid_cs] / denominator2[valid_cs]

        ssim_map = torch.ones_like(mu_image)
        valid_ssim = denominator1 * denominator2 > 0
        ssim_map[valid_ssim] = (
            numerator1[valid_ssim] * numerator2[valid_ssim] / (denominator1[valid_ssim] * denominator2[valid_ssim])
        )

        fallback_ssim = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[fallback_ssim] = numerator1[fallback_ssim] / denominator1[fallback_ssim]

    return ssim_map, contrast_structure_map
