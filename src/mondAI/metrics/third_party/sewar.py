from collections.abc import Callable

import torch


def get_sewar_mse() -> Callable[..., torch.Tensor]:
    from sewar.full_ref import mse as mse_sewar

    def sewar_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # sewar operates on NumPy arrays.
        value = mse_sewar(reference.cpu().numpy(), image.cpu().numpy())
        return torch.tensor(value, device=image.device, dtype=image.dtype)

    return sewar_mse


def get_sewar_rmse() -> Callable[..., torch.Tensor]:
    from sewar.full_ref import rmse as rmse_sewar

    def sewar_rmse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # sewar operates on NumPy arrays.
        value = rmse_sewar(reference.cpu().numpy(), image.cpu().numpy())
        return torch.tensor(value, device=image.device, dtype=image.dtype)

    return sewar_rmse


def get_sewar_msssim(
    weights: tuple[float, ...], ws: int, K1: float, K2: float, MAX: float
) -> Callable[..., torch.Tensor]:
    from sewar.full_ref import msssim as msssim_sewar

    def sewar_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # sewar operates on NumPy arrays and follows a classic image-quality API.
        score = msssim_sewar(
            reference.cpu().numpy(),
            image.cpu().numpy(),
            weights=weights,
            ws=ws,
            K1=K1,
            K2=K2,
            MAX=MAX,
        )
        return torch.tensor(score, device=image.device, dtype=image.dtype)

    return sewar_msssim


def get_sewar_psnr(MAX: float) -> Callable[..., torch.Tensor]:
    from sewar.full_ref import psnr as psnr_sewar

    def sewar_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # sewar operates on NumPy arrays.
        value = psnr_sewar(reference.cpu().numpy(), image.cpu().numpy(), MAX=MAX)
        return torch.tensor(value, device=image.device, dtype=image.dtype)

    return sewar_psnr


def get_sewar_ssim(ws: int, K1: float, K2: float, MAX: float, sigma: float) -> Callable[..., torch.Tensor]:
    from sewar.full_ref import ssim as ssim_sewar
    from sewar.utils import Filter

    def sewar_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # sewar operates on NumPy arrays and exposes filter settings explicitly.
        # By default, it uses a uniform filter, so this must be overridden.
        # Using ``mode="valid"`` keeps it aligned with the original MATLAB
        # implementation.
        score, _ = ssim_sewar(
            reference.cpu().numpy(),
            image.cpu().numpy(),
            ws=ws,
            K1=K1,
            K2=K2,
            MAX=MAX,
            fltr_specs={"fltr": Filter.GAUSSIAN, "sigma": sigma, "ws": ws},
            mode="valid",
        )
        return torch.tensor(score, device=image.device, dtype=image.dtype)

    return sewar_ssim


def get_sewar_vifp(sigma_n_sq: float) -> Callable[..., torch.Tensor]:
    from sewar.full_ref import vifp as vifp_sewar

    def sewar_vifp(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # sewar's implementation expects numpy arrays

        vifp_numpy = vifp_sewar(image.cpu().numpy(), reference.cpu().numpy(), sigma_nsq=sigma_n_sq)
        return torch.tensor(vifp_numpy).to(image.device).to(image.dtype)

    return sewar_vifp
