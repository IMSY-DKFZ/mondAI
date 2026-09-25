from collections.abc import Callable, Sequence

import torch


def get_monai_mae() -> Callable[..., torch.Tensor]:
    from monai.metrics import MAEMetric

    def monai_mae(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        mae_metric = MAEMetric()
        mae_value = mae_metric(image.unsqueeze(0), reference.unsqueeze(0))
        return torch.as_tensor(mae_value, device=image.device)

    return monai_mae


def get_monai_mse() -> Callable[..., torch.Tensor]:
    from monai.metrics import MSEMetric

    def monai_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        mse_metric = MSEMetric()
        mse_value = mse_metric(image.unsqueeze(0), reference.unsqueeze(0))
        return torch.as_tensor(mse_value, device=image.device)

    return monai_mse


def get_monai_rmse() -> Callable[..., torch.Tensor]:
    from monai.metrics import RMSEMetric

    def monai_rmse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        rmse_metric = RMSEMetric()
        rmse_value = rmse_metric(image.unsqueeze(0), reference.unsqueeze(0))
        return torch.as_tensor(rmse_value, device=image.device)

    return monai_rmse


def get_monai_msssim(
    data_range: float, kernel_size: int, kernel_sigma: float, weights: Sequence[float], k1: float, k2: float
) -> Callable[..., torch.Tensor]:
    from monai.metrics import MultiScaleSSIMMetric

    monai_metric = MultiScaleSSIMMetric(
        spatial_dims=2,
        data_range=data_range,
        kernel_type="gaussian",
        kernel_size=kernel_size,
        kernel_sigma=kernel_sigma,
        weights=weights,
        k1=k1,
        k2=k2,
    )

    def monai_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # MONAI exposes MS-SSIM as a metric object for batched medical imaging.
        return monai_metric(reference.unsqueeze(0).unsqueeze(0), image.unsqueeze(0).unsqueeze(0))

    return monai_msssim


def get_monai_psnr(max_val: float) -> Callable[..., torch.Tensor]:
    from monai.metrics import PSNRMetric

    monai_metric = PSNRMetric(max_val=max_val)

    def monai_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # MONAI expects tensors in NCHW format.
        return torch.as_tensor(
            monai_metric(image.unsqueeze(0).unsqueeze(0), reference.unsqueeze(0).unsqueeze(0)),
            device=image.device,
        )

    return monai_psnr


def get_monai_ssim(
    data_range: float, win_size: int, kernel_sigma: float, k1: float, k2: float
) -> Callable[..., torch.Tensor]:
    from monai.metrics import SSIMMetric

    monai_metric = SSIMMetric(
        spatial_dims=2,
        data_range=data_range,
        kernel_type="gaussian",
        win_size=win_size,
        kernel_sigma=kernel_sigma,
        k1=k1,
        k2=k2,
    )

    def monai_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # MONAI exposes SSIM as a metric object and expects NCHW tensors.
        # It uses valid padding, source of differences still unclear.
        return monai_metric(reference.unsqueeze(0).unsqueeze(0), image.unsqueeze(0).unsqueeze(0))

    return monai_ssim
