from collections.abc import Callable

import torch


def get_piq_dss(sigma_weight: float) -> Callable[..., torch.Tensor]:
    from piq import dss

    def piq_dss(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # PIQ expects NCHW tensors and provides a PyTorch-native DSS implementation.
        return dss(
            image.unsqueeze(0).unsqueeze(0),
            reference.unsqueeze(0).unsqueeze(0),
            data_range=255.0,
            sigma_weight=1.55,
        )

    return piq_dss


def get_piq_fsim(
    use_rgb: bool,
    scales: int,
    orientations: int,
    min_length: int,
    mult: int,
    sigma_f: float,
    delta_theta: float,
    k: float,
) -> Callable[..., torch.Tensor]:
    from piq import fsim

    def piq_fsim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # piq's implementation expects inputs with shape (N, C, H, W) and
        # data_range parameter should correspond to pixel value range

        if not use_rgb:
            image = image.unsqueeze(0)
            reference = reference.unsqueeze(0)

        return fsim(
            image.unsqueeze(0),
            reference.unsqueeze(0),
            data_range=255.0,
            chromatic=use_rgb,
            scales=scales,
            orientations=orientations,
            min_length=min_length,
            mult=mult,
            sigma_f=sigma_f,
            delta_theta=delta_theta,
            k=k,
        )

    return piq_fsim


def get_piq_gmsd(t: float) -> Callable[..., torch.Tensor]:
    from piq import gmsd

    def piq_gmsd(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # PIQ expects NCHW tensors and provides a PyTorch-native GMSD implementation.
        return gmsd(
            image.unsqueeze(0).unsqueeze(0),
            reference.unsqueeze(0).unsqueeze(0),
            data_range=255.0,
            t=t,
        )

    return piq_gmsd


def get_piq_haarpsi(use_rgb: bool, c: float, alpha: float, subsample: bool) -> Callable[..., torch.Tensor]:
    from piq import haarpsi

    def piq_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # piq's implementation expects inputs with shape (N, C, H, W) and
        # data_range parameter should correspond to pixel value range

        if not use_rgb:
            image = image.unsqueeze(0)
            reference = reference.unsqueeze(0)

        return haarpsi(
            image.unsqueeze(0),
            reference.unsqueeze(0),
            data_range=255.0,
            c=c,
            alpha=alpha,
            subsample=subsample,
        )

    return piq_haarpsi


def get_piq_iwssim(
    data_range: float,
    kernel_size: int,
    kernel_sigma: float,
    weights: tuple[float, ...],
    k1: float,
    k2: float,
    parent: bool,
    blk_size: int,
    sigma_nsq: float,
) -> Callable[..., torch.Tensor]:
    from piq import information_weighted_ssim

    def piq_iwssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # PIQ expects NCHW tensors and provides a PyTorch-native IW-SSIM
        # implementation that closely matches the original formulation.
        return information_weighted_ssim(
            image.unsqueeze(0).unsqueeze(0),
            reference.unsqueeze(0).unsqueeze(0),
            data_range=data_range,
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            scale_weights=torch.tensor(weights, device=image.device, dtype=image.dtype),
            k1=k1,
            k2=k2,
            parent=parent,
            blk_size=blk_size,
            sigma_nsq=sigma_nsq,
        )

    return piq_iwssim


def get_piq_mdsi(combination: str) -> Callable[..., torch.Tensor]:
    from piq import mdsi

    def piq_mdsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # PIQ expects NCHW tensors and provides a PyTorch-native MDSI implementation.
        return mdsi(
            image.unsqueeze(0),
            reference.unsqueeze(0),
            data_range=255.0,
            combination="mult" if combination == "product" else combination,
        )

    return piq_mdsi


def get_piq_msssim(
    kernel_size: int, kernel_sigma: float, data_range: float, scale_weights: tuple[float, ...], k1: float, k2: float
) -> Callable[..., torch.Tensor]:
    from piq import multi_scale_ssim

    def piq_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # PIQ expects NCHW tensors and exposes the number of scales through the weights.
        return multi_scale_ssim(
            image.unsqueeze(0).unsqueeze(0),
            reference.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            data_range=data_range,
            reduction="mean",
            scale_weights=torch.tensor(scale_weights, device=image.device, dtype=image.dtype),
            k1=k1,
            k2=k2,
        )

    return piq_msssim


def get_piq_psnr(data_range: float) -> Callable[..., torch.Tensor]:
    from piq import psnr

    def piq_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # PIQ expects NCHW tensors.
        return psnr(
            image.unsqueeze(0).unsqueeze(0),
            reference.unsqueeze(0).unsqueeze(0),
            data_range=data_range,
            reduction="mean",
            convert_to_greyscale=False,
        )

    return piq_psnr


def get_piq_ssim(
    kernel_size: int, kernel_sigma: float, data_range: float, k1: float, k2: float
) -> Callable[..., torch.Tensor]:
    from piq import ssim as ssim_piq

    def piq_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # PIQ expects NCHW tensors. In practice, its results are typically
        # very close to scikit-image, with only minor differences stemming
        # from implementation details such as kernel construction.
        return ssim_piq(
            image.unsqueeze(0).unsqueeze(0),
            reference.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            data_range=data_range,
            reduction="mean",
            full=False,
            downsample=False,
            k1=k1,
            k2=k2,
        )

    return piq_ssim


def get_piq_vifp(sigma_n_sq: float) -> Callable[..., torch.Tensor]:
    from piq import vif_p as vifp_piq

    def piq_vifp(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # piq's implementation expects inputs with shape (N, C, H, W) and
        # data_range parameter should correspond to pixel value range

        return vifp_piq(
            image.unsqueeze(0).unsqueeze(0),
            reference.unsqueeze(0).unsqueeze(0),
            data_range=255.0,
            sigma_n_sq=sigma_n_sq,
        )

    return piq_vifp


def get_piq_vsi(c1: float, c2: float, c3: float, alpha: float, beta: float) -> Callable[..., torch.Tensor]:
    from piq import vsi

    def piq_vsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # PIQ expects NCHW tensors and provides a PyTorch-native VSI implementation.
        return vsi(
            image.unsqueeze(0), reference.unsqueeze(0), data_range=255.0, c1=c1, c2=c2, c3=c3, alpha=alpha, beta=beta
        )

    return piq_vsi
