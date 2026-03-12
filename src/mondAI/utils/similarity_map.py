import torch


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
