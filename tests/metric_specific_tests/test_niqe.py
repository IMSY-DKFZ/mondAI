import pytest
import torch

from mondAI.metrics.no_reference.niqe import NIQE


@pytest.mark.parametrize("factor", [1.1, -1.0])
def test_invalid_value_range_image(factor: float) -> None:
    with pytest.raises(ValueError):
        niqe = NIQE()
        img1 = torch.rand(200, 300) * 255.0 * factor
        niqe(img1)


def test_too_small_image() -> None:
    with pytest.raises(ValueError):
        niqe = NIQE()
        img1 = torch.ones(64, 64) * 255.0
        niqe(img1)


@pytest.mark.parametrize(
    "parameter, values",
    [
        ("block_size_row", [96, 42, 25]),
        ("block_size_column", [96, 42, 25]),
        ("block_column_overlap", [0, 1, 2]),
        ("block_row_overlap", [0, 1, 2]),
    ],
)
def test_parameter_valid(
    parameter: str,
    values: list[int],
) -> None:
    for value in values:
        kwargs = {parameter: value}
        niqe = NIQE(**kwargs)
        img1 = torch.rand(200, 300) * 255.0
        niqe(img1)


@pytest.mark.parametrize(
    "parameter, values",
    [
        ("block_size_row", [0, -1]),
        ("block_size_column", [0, -1]),
        ("block_column_overlap", [-1]),
        ("block_row_overlap", [-1]),
    ],
)
def test_parameter_invalid(
    parameter: str,
    values: list[int],
) -> None:
    for value in values:
        kwargs = {parameter: value}
        with pytest.raises(ValueError):
            NIQE(**kwargs)
