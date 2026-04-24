from collections.abc import Callable

import torch


def get_piqa_fsim(
    use_rgb: bool, t1: float, t2: float, t3: float, t4: float, lmbda: float
) -> Callable[..., torch.Tensor]:
    from piqa.fsim import fsim as fsim_piqa
    from piqa.fsim import gradient_kernel, pc_filters, phase_congruency, scharr_kernel

    def piqa_fsim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # piqa's implementation expects inputs with shape (N, C, H, W) and
        # data_range parameter should correspond to pixel value range

        image = image.unsqueeze(0)
        reference = reference.unsqueeze(0)

        if not use_rgb:
            image = image.unsqueeze(0)
            reference = reference.unsqueeze(0)

        filters_1 = pc_filters(image)
        filters_2 = pc_filters(reference)
        pc_1 = phase_congruency(image[:, :1, :, :] if image.shape[1] == 3 else image, filters_1)
        pc_2 = phase_congruency(reference[:, :1, :, :] if reference.shape[1] == 3 else reference, filters_2)
        kernel = gradient_kernel(scharr_kernel().to(image.device))

        return fsim_piqa(
            image.float() * 255.0,
            reference.float() * 255.0,
            pc_1,
            pc_2,
            kernel,
            value_range=255.0,
            t1=t1,
            t2=t2,
            t3=t3,
            t4=t4,
            lmbda=lmbda,
        )

    return piqa_fsim


def get_piqa_haarpsi(use_rgb: bool, c: float, alpha: float) -> Callable[..., torch.Tensor]:
    from piqa.haarpsi import haarpsi as haarpsi_piqa

    def piqa_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # piqa's implementation expects inputs with shape (N, C, H, W) and
        # data_range parameter should correspond to pixel value range

        if not use_rgb:
            image = image.unsqueeze(0)
            reference = reference.unsqueeze(0)

        return haarpsi_piqa(
            image.unsqueeze(0).float() / 255.0,
            reference.unsqueeze(0).float() / 255.0,
            value_range=1.0,
            c=c,
            alpha=alpha,
        )

    return piqa_haarpsi


def get_piqa_msssim(
    window_size: int, sigma: float, value_range: float, weights: tuple[float, ...], k1: float, k2: float
) -> Callable[..., torch.Tensor]:
    from piqa import MS_SSIM

    def piqa_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # PIQA exposes MS-SSIM as a module and expects NCHW tensors.
        piqa_metric = MS_SSIM(
            window_size=window_size,
            sigma=sigma,
            n_channels=1,
            reduction="mean",
            value_range=value_range,
            weights=torch.tensor(weights, device=image.device, dtype=image.dtype),
            k1=k1,
            k2=k2,
        ).to(image.device)

        return piqa_metric(image.float().unsqueeze(0).unsqueeze(0), reference.float().unsqueeze(0).unsqueeze(0))

    return piqa_msssim


def get_piqa_psnr(value_range: float) -> Callable[..., torch.Tensor]:
    from piqa import PSNR as PIQAPSNR

    piqa_metric = PIQAPSNR(value_range=value_range, reduction="mean")

    def piqa_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # PIQA exposes PSNR as a module and expects NCHW tensors.
        return piqa_metric(image.unsqueeze(0).unsqueeze(0), reference.unsqueeze(0).unsqueeze(0))

    return piqa_psnr


def get_piqa_ssim(
    window_size: int, sigma: float, value_range: float, k1: float, k2: float
) -> Callable[..., torch.Tensor]:
    from piqa import SSIM as PIQASSIM

    piqa_metric = PIQASSIM(
        window_size=window_size,
        sigma=sigma,
        n_channels=1,
        reduction="mean",
        value_range=value_range,
        k1=k1,
        k2=k2,
    )

    def piqa_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # PIQA exposes SSIM as a module. It expects NCHW tensors and an
        # explicit channel count. In practice, it behaves very similarly to
        # scikit-image for the parameter settings used here.
        piqa_metric.to(image.device)
        return piqa_metric(image.float().unsqueeze(0).unsqueeze(0), reference.float().unsqueeze(0).unsqueeze(0))

    return piqa_ssim
