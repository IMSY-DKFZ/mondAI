from collections.abc import Callable

import torch


def get_skimage_mse() -> Callable[..., torch.Tensor]:
    from skimage.metrics import mean_squared_error as mse_skimage

    def skimage_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Flatten the tensors and convert to numpy arrays for scikit-image
        image_np = image.cpu().numpy().flatten()
        reference_np = reference.cpu().numpy().flatten()
        mse_value = mse_skimage(reference_np, image_np)
        return torch.tensor(mse_value, device=image.device)

    return skimage_mse


def get_skimage_psnr(data_range: float) -> Callable[..., torch.Tensor]:
    from skimage.metrics import peak_signal_noise_ratio as psnr_skimage

    def skimage_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # scikit-image operates on NumPy arrays and is the closest API match.
        value = psnr_skimage(
            reference.cpu().numpy(),
            image.cpu().numpy(),
            data_range=data_range,
        )
        return torch.tensor(value, device=image.device, dtype=image.dtype)

    return skimage_psnr


def get_skimage_ssim(
    win_size: int, data_range: float, K1: float, K2: float, sigma: float
) -> Callable[..., torch.Tensor]:
    from skimage.metrics import structural_similarity

    def skimage_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # scikit-image is the closest direct reference implementation.
        # Differences relative to mondAI are usually limited to floating-point
        # precision, especially when operating on float32 inputs.
        score = structural_similarity(
            reference.cpu().numpy(),
            image.cpu().numpy(),
            win_size=win_size,
            gradient=False,
            data_range=data_range,
            channel_axis=None,
            gaussian_weights=True,
            full=False,
            use_sample_covariance=False,
            K1=K1,
            K2=K2,
            sigma=sigma,
        )
        return torch.tensor(score, device=image.device, dtype=image.dtype)

    return skimage_ssim
