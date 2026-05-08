from collections.abc import Callable

import torch


def get_medimetrics_cwssim() -> Callable[..., torch.Tensor]:
    from mondAI.metrics.third_party.medimetrics_code.cwssim import CWSSIM as MediMetricsCWSSIM

    metric_medimetrics = MediMetricsCWSSIM()

    def medimetrics_cwssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # medimetrics provides a CW-SSIM implementation based on
        # code from the IQA_pytorch package, which is itself based on the original matlab code
        # but they only used an approximation for the complex steerable pyramid decomposition
        # and other default parameters, 8 orientations and 4 levels
        score = metric_medimetrics.compute(reference.cpu().numpy(), image.cpu().numpy())
        return torch.tensor(score, device=image.device, dtype=image.dtype)

    return medimetrics_cwssim


def get_medimetrics_mae() -> Callable[..., torch.Tensor]:
    from mondAI.metrics.third_party.medimetrics_code.mae import MAE as MAEMedimetrics

    def medimetric_mae(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        medimetrics_metric = MAEMedimetrics()
        mae_value = medimetrics_metric.compute(image.numpy(force=True), reference.numpy(force=True))
        return torch.as_tensor(mae_value, device=image.device)

    return medimetric_mae


def get_medimetrics_mse() -> Callable[..., torch.Tensor]:
    from mondAI.metrics.third_party.medimetrics_code.mse import MSE as MSEMedimetrics

    def medimetric_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        medimetrics_metric = MSEMedimetrics()
        mse_value = medimetrics_metric.compute(image.numpy(force=True), reference.numpy(force=True))
        return torch.as_tensor(mse_value, device=image.device)

    return medimetric_mse


def get_medimetrics_msssim(
    data_range: float, kernel_size: int, k1: float, k2: float, sigma: float, betas: tuple[float, ...]
) -> Callable[..., torch.Tensor]:
    from mondAI.metrics.third_party.medimetrics_code.ssim import MSSSIM as MediMetricsMSSSIM

    metric = MediMetricsMSSSIM()

    def medimetrics_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # medimetrics exposes MS-SSIM through the SSIM metric class using a
        # NumPy-based compute method for medical image quality assessment.
        score = metric.compute(
            reference.cpu().numpy(),
            image.cpu().numpy(),
            data_range=data_range,
            kernel_size=kernel_size,
            k1=k1,
            k2=k2,
            sigma=sigma,
            betas=list(betas),
        )
        return torch.tensor(score, device=image.device, dtype=image.dtype)

    return medimetrics_msssim


def get_medimetrics_niqe() -> Callable[..., torch.Tensor]:
    from mondAI.metrics.third_party.medimetrics_code.niqe import NIQE as MediMetricsNIQE

    medimetric = MediMetricsNIQE()

    def medimetrics_niqe(image: torch.Tensor) -> torch.Tensor:
        # medimetrics exposes a NumPy-based compute method.
        value = medimetric.compute(image.cpu().numpy(), data_range=255)
        return torch.tensor(value, device=image.device, dtype=image.dtype)

    return medimetrics_niqe


def get_medimetrics_psnr(data_range: float) -> Callable[..., torch.Tensor]:
    from mondAI.metrics.third_party.medimetrics_code.psnr import PSNR as MediMetricsPSNR

    medimetric = MediMetricsPSNR()

    def medimetrics_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # medimetrics exposes a NumPy-based compute method.
        value = medimetric.compute(reference.cpu().numpy(), image.cpu().numpy(), data_range=data_range)
        return torch.tensor(value, device=image.device, dtype=image.dtype)

    return medimetrics_psnr


def get_medimetrics_ssim(
    data_range: float, kernel_size: int, sigma: float, k1: float, k2: float
) -> Callable[..., torch.Tensor]:
    from mondAI.metrics.third_party.medimetrics_code.ssim import SSIM as MediMetricsSSIM

    medimetric = MediMetricsSSIM()

    def medimetrics_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        score = medimetric.compute(
            reference.cpu().numpy(),
            image.cpu().numpy(),
            data_range=data_range,
            kernel_size=kernel_size,
            sigma=sigma,
            k1=k1,
            k2=k2,
        )
        return torch.tensor(score, device=image.device, dtype=image.dtype)

    return medimetrics_ssim
