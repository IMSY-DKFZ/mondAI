import pytest
import torch

from mondAI.metrics.full_reference.cwssim import CWSSIM


@pytest.mark.parametrize("levels", [1, 2, 4])
def test_levels_valid(levels: int) -> None:
    metric = CWSSIM(levels=levels)
    img1 = torch.rand(400, 400)
    img2 = torch.rand(400, 400)
    metric(img1, img2)


@pytest.mark.parametrize("levels", [0, -1])
def test_levels_invalid(levels: int) -> None:
    with pytest.raises(ValueError):
        CWSSIM(levels=levels)


@pytest.mark.parametrize("orientations", [1, 4, 8])
def test_orientations_valid(orientations: int) -> None:
    metric = CWSSIM(orientations=orientations)
    img1 = torch.rand(400, 400)
    img2 = torch.rand(400, 400)
    metric(img1, img2)


@pytest.mark.parametrize("orientations", [0, -1])
def test_orientations_invalid(orientations: int) -> None:
    with pytest.raises(ValueError):
        CWSSIM(orientations=orientations)


@pytest.mark.parametrize("guard_boundary", [0, 1, 2])
def test_guard_boundary_valid(guard_boundary: int) -> None:
    metric = CWSSIM(guard_boundary=guard_boundary)
    img1 = torch.rand(400, 400)
    img2 = torch.rand(400, 400)
    metric(img1, img2)


def test_guard_boundary_invalid() -> None:
    with pytest.raises(ValueError):
        CWSSIM(guard_boundary=-1)


@pytest.mark.parametrize("k", [0.0, 1e-6, 0.1])
def test_k_valid(k: float) -> None:
    metric = CWSSIM(k=k)
    img1 = torch.rand(400, 400)
    img2 = torch.rand(400, 400)
    metric(img1, img2)


def test_k_invalid() -> None:
    with pytest.raises(ValueError):
        CWSSIM(k=-1e-6)


def test_image_too_small() -> None:
    metric = CWSSIM()
    img1 = torch.ones(128, 128)
    img2 = torch.ones(128, 128)
    with pytest.raises(ValueError):
        metric(img1, img2)


def test_identical_images_are_one() -> None:
    metric = CWSSIM()
    img = torch.rand(400, 400)
    result = metric(img, img)
    assert torch.isclose(torch.as_tensor(result), torch.tensor(1.0, dtype=torch.as_tensor(result).dtype), atol=1e-5)
