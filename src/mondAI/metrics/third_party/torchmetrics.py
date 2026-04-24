from collections.abc import Callable

import torch


def get_torchmetrics_mse() -> Callable[..., torch.Tensor]:
    from torchmetrics.functional.regression import mean_squared_error as mse_torchmetrics

    def torchmetrics_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return mse_torchmetrics(image.contiguous(), reference.contiguous())

    return torchmetrics_mse


def get_torchmetrics_msssim(
    sigma: float, kernel_size: int, data_range: float, k1: float, k2: float, betas: tuple[float, ...]
) -> Callable[..., torch.Tensor]:
    from torchmetrics.functional.image import multiscale_structural_similarity_index_measure

    def torchmetrics_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # torchmetrics expects NCHW tensors and implements the multiscale
        # aggregation internally.
        return multiscale_structural_similarity_index_measure(
            image.unsqueeze(0).unsqueeze(0),
            reference.unsqueeze(0).unsqueeze(0),
            gaussian_kernel=True,
            sigma=sigma,
            kernel_size=kernel_size,
            reduction="elementwise_mean",
            data_range=data_range,
            k1=k1,
            k2=k2,
            betas=betas,
            normalize=None,
        )

    return torchmetrics_msssim


def get_torchmetrics_psnr(data_range: float) -> Callable[..., torch.Tensor]:
    from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_torchmetrics

    def torchmetrics_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # torchmetrics expects NCHW tensors for image metrics.
        return psnr_torchmetrics(
            image.unsqueeze(0).unsqueeze(0),
            reference.unsqueeze(0).unsqueeze(0),
            data_range=data_range,
        )

    return torchmetrics_psnr


def get_torchmetrics_ssim(
    sigma: float, kernel_size: int, data_range: float, k1: float, k2: float
) -> Callable[..., torch.Tensor]:
    from torchmetrics.functional.image import structural_similarity_index_measure

    def torchmetrics_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # torchmetrics expects NCHW tensors. Its implementation differs from
        # the original MATLAB code, for example by using reflection padding
        # instead of strict valid convolution.
        return structural_similarity_index_measure(
            image.unsqueeze(0).unsqueeze(0),
            reference.unsqueeze(0).unsqueeze(0),
            gaussian_kernel=True,
            sigma=sigma,
            kernel_size=kernel_size,
            reduction="elementwise_mean",
            data_range=data_range,
            k1=k1,
            k2=k2,
            return_full_image=False,
            return_contrast_sensitivity=False,
        )

    return torchmetrics_ssim


def get_torchmetrics_vifp(sigma_n_sq: float) -> Callable[..., torch.Tensor]:
    from torchmetrics.functional.image.vif import visual_information_fidelity as vifp_torchmetrics

    def torchmetrics_vifp(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # torchmetrics' implementation expects inputs with shape (N, C, H, W)

        return vifp_torchmetrics(
            image.unsqueeze(0).unsqueeze(0),
            reference.unsqueeze(0).unsqueeze(0),
            sigma_n_sq=sigma_n_sq,
        )

    return torchmetrics_vifp
