from collections.abc import Callable

import torch


def get_deepinv_haarpsi(
    use_rgb: bool, C: float, alpha: float, preprocess_with_subsampling: bool
) -> Callable[..., torch.Tensor]:
    from deepinv.loss.metric import HaarPSI as DeepInvHaarPSI

    def deepinv_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # DeepInv's implementation expects inputs with shape (N, C, H, W) and
        #  pixel values in [0, 1] with single precision (float32, therefore scores might deviate slightly)

        if not use_rgb:
            image = image.unsqueeze(0)
            reference = reference.unsqueeze(0)

        haarpsi_metric = DeepInvHaarPSI(C=C, alpha=alpha, preprocess_with_subsampling=preprocess_with_subsampling)
        return haarpsi_metric(
            image.unsqueeze(0).float() / 255.0,
            reference.unsqueeze(0).float() / 255.0,
        )

    return deepinv_haarpsi


def get_deepinv_mse() -> Callable[..., torch.Tensor]:
    from deepinv.loss.metric import MSE as MSEDeepInv

    def deepinv_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        mse_metric = MSEDeepInv()
        mse_value = mse_metric(image.unsqueeze(0).unsqueeze(0), reference.unsqueeze(0).unsqueeze(0))
        return torch.as_tensor(mse_value, device=image.device)

    return deepinv_mse


def get_deepinv_psnr(max_pixel: float) -> Callable[..., torch.Tensor]:
    from deepinv.loss.metric import PSNR as DeepInvPSNR

    deepinv_metric = DeepInvPSNR(max_pixel=max_pixel)

    def deepinv_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # deepinv expects NCHW tensors.
        return torch.as_tensor(
            deepinv_metric(
                image.unsqueeze(0).unsqueeze(0),
                reference.unsqueeze(0).unsqueeze(0),
            ),
            device=image.device,
        )

    return deepinv_psnr


def get_deepinv_ssim(
    max_pixel: float, sigma: float, kernel_size: int, k1: float, k2: float
) -> Callable[..., torch.Tensor]:
    from deepinv.loss.metric import SSIM as DeepInvSSIM

    deepinv_metric = DeepInvSSIM(
        multiscale=False,
        max_pixel=max_pixel,
        min_pixel=0.0,
        torchmetric_kwargs={
            "gaussian_kernel": True,
            "sigma": sigma,
            "kernel_size": kernel_size,
            "k1": k1,
            "k2": k2,
        },
    )

    def deepinv_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # deepinv wraps torchmetrics-style SSIM for inverse-problems workflows,
        # so its results are expected to match torchmetrics very closely.
        return deepinv_metric(image.unsqueeze(0).unsqueeze(0), reference.unsqueeze(0).unsqueeze(0))

    return deepinv_ssim
