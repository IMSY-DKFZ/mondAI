import matplotlib.pyplot as plt
import numpy as np
import torch
from pytest import fixture
from skimage.data import brain as brain_volume
from skimage.data import shepp_logan_phantom


def load_shepp_logan_phantom() -> np.ndarray:
    # Load the Shepp-Logan phantom (400, 400) pixel image
    return shepp_logan_phantom()


def load_brain() -> np.ndarray:
    # Load the brain volume (10, 256, 256) voxel image
    # TODO: Should this volume be normalized to [0, 1]?
    # TODO: Does this need to be converted from uint8 to float64?
    return brain_volume()


def create_random_image(shape: tuple[int, ...], dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Create a random image tensor with the specified shape and data type."""
    return torch.rand(shape, dtype=dtype)


@fixture
def phantom() -> np.ndarray:
    return load_shepp_logan_phantom()


@fixture
def brain() -> np.ndarray:
    return load_brain()


@fixture
def brain_slice() -> np.ndarray:
    return load_brain()[4]  # Return the middle slice of the brain volume


@fixture
def grayscale_image_W_H() -> torch.Tensor:
    return create_random_image((256, 256))


@fixture
def grayscale_image_W_H_1() -> torch.Tensor:
    return create_random_image((256, 256, 1))


@fixture
def rgb_image_C_W_H() -> torch.Tensor:
    return create_random_image((3, 256, 256))


@fixture
def rgb_image_W_H_C() -> torch.Tensor:
    return create_random_image((256, 256, 3))


@fixture
def multispectral_image_C_W_H() -> torch.Tensor:
    return create_random_image((10, 256, 256))


@fixture
def multispectral_image_W_H_C() -> torch.Tensor:
    return create_random_image((256, 256, 10))


@fixture
def batch_of_grayscale_images_B_C_W_H() -> torch.Tensor:
    return create_random_image((4, 1, 256, 256))


@fixture
def batch_of_rgb_images_B_C_W_H() -> torch.Tensor:
    return create_random_image((4, 3, 256, 256))


@fixture
def batch_of_multispectral_images_B_C_W_H() -> torch.Tensor:
    return create_random_image((4, 10, 256, 256))


@fixture
def batch_of_grayscale_images_B_W_H() -> torch.Tensor:
    return create_random_image((4, 256, 256))


@fixture
def batch_of_rgb_images_B_W_H_C() -> torch.Tensor:
    return create_random_image((4, 256, 256, 3))


@fixture
def batch_of_multispectral_images_B_W_H_C() -> torch.Tensor:
    return create_random_image((4, 256, 256, 10))


@fixture
def batch_of_grayscale_images_B_W_H_1() -> torch.Tensor:
    return create_random_image((4, 256, 256, 1))


@fixture
def batch_of_rgb_images_B_W_H_3() -> torch.Tensor:
    return create_random_image((4, 256, 256, 3))


@fixture
def batch_of_multispectral_images_B_W_H_10() -> torch.Tensor:
    return create_random_image((4, 256, 256, 10))


@fixture
def grayscale_volume_D_W_H() -> torch.Tensor:
    return create_random_image((10, 256, 256))


@fixture
def grayscale_volume_D_W_H_1() -> torch.Tensor:
    return create_random_image((10, 256, 256, 1))


@fixture
def rgb_volume_D_C_W_H() -> torch.Tensor:
    return create_random_image((10, 3, 256, 256))


@fixture
def rgb_volume_D_W_H_C() -> torch.Tensor:
    return create_random_image((10, 256, 256, 3))


@fixture
def multispectral_volume_D_C_W_H() -> torch.Tensor:
    return create_random_image((10, 10, 256, 256))


@fixture
def multispectral_volume_D_W_H_C() -> torch.Tensor:
    return create_random_image((10, 256, 256, 10))


@fixture
def batch_of_grayscale_volumes_B_D_W_H() -> torch.Tensor:
    return create_random_image((4, 10, 256, 256))


@fixture
def batch_of_rgb_volumes_B_D_C_W_H() -> torch.Tensor:
    return create_random_image((4, 10, 3, 256, 256))


@fixture
def batch_of_rgb_volumes_B_D_W_H_C() -> torch.Tensor:
    return create_random_image((4, 10, 256, 256, 3))


@fixture
def batch_of_multispectral_volumes_B_D_C_W_H() -> torch.Tensor:
    return create_random_image((4, 10, 10, 256, 256))


@fixture
def batch_of_multispectral_volumes_B_D_W_H_C() -> torch.Tensor:
    return create_random_image((4, 10, 256, 256, 10))


# Example usage
if __name__ == "__main__":
    phantom = load_shepp_logan_phantom()
    print("Shepp-Logan Phantom shape:", phantom.shape, type(phantom))

    brain = load_brain()
    print("Brain Volume shape:", brain.shape, type(brain))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Shepp-Logan Phantom")
    plt.imshow(phantom, cmap="gray")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Brain")
    plt.imshow(brain[4], cmap="gray")
    plt.colorbar()
    plt.show()
