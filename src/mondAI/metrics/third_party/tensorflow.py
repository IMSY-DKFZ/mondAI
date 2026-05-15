from collections.abc import Callable

import torch


def get_tensorflow_mae() -> Callable[..., torch.Tensor]:
    import tensorflow as tf

    def tensorflow_mae(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Flatten the tensors and convert to numpy arrays for TensorFlow

        image_np = image.cpu().numpy()
        reference_np = reference.cpu().numpy()
        mae_metric = tf.keras.metrics.MeanAbsoluteError()
        mae_metric.update_state(reference_np, image_np)
        mae_value = mae_metric.result().numpy()
        return torch.as_tensor(mae_value, device=image.device)

    return tensorflow_mae


def get_tensorflow_mse() -> Callable[..., torch.Tensor]:
    import tensorflow as tf

    def tensorflow_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        image_np = image.detach().cpu().numpy()
        reference_np = reference.detach().cpu().numpy()
        mse_metric = tf.keras.metrics.MeanSquaredError()
        mse_metric.update_state(reference_np, image_np)
        mse_value = mse_metric.result().numpy()
        return torch.as_tensor(mse_value, device=image.device)

    return tensorflow_mse


def get_tensorflow_rmse() -> Callable[..., torch.Tensor]:
    import tensorflow as tf

    def tensorflow_rmse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        image_np = image.detach().cpu().numpy()
        reference_np = reference.detach().cpu().numpy()
        rmse_metric = tf.keras.metrics.RootMeanSquaredError()
        rmse_metric.update_state(reference_np, image_np)
        rmse_value = rmse_metric.result().numpy()
        return torch.as_tensor(rmse_value, device=image.device)

    return tensorflow_rmse


def get_tensorflow_msssim(
    max_val: float, power_factors: tuple[float, ...], filter_size: int, filter_sigma: float, k1: float, k2: float
) -> Callable[..., torch.Tensor]:
    import tensorflow as tf

    def tensorflow_msssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # TensorFlow expects NHWC tensors and supports configurable scale weights.
        score = tf.image.ssim_multiscale(
            image.cpu().numpy()[None, ..., None],
            reference.cpu().numpy()[None, ..., None],
            max_val=max_val,
            power_factors=power_factors,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
        ).numpy()
        return torch.tensor(score, device=image.device, dtype=image.dtype)

    return tensorflow_msssim


def get_tensorflow_psnr(max_val: float) -> Callable[..., torch.Tensor]:
    import tensorflow as tf

    def tensorflow_psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # TensorFlow expects NHWC tensors.
        value = tf.image.psnr(
            image.cpu().numpy()[None, ..., None],
            reference.cpu().numpy()[None, ..., None],
            max_val=max_val,
        ).numpy()
        return torch.tensor(value.item(), device=image.device, dtype=image.dtype)

    return tensorflow_psnr


def get_tensorflow_ssim(
    max_val: float, filter_size: int, filter_sigma: float, k1: float, k2: float
) -> Callable[..., torch.Tensor]:
    import tensorflow as tf

    def tensorflow_ssim(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # TensorFlow expects NHWC tensors. Small deviations may arise from
        # its internal use of single precision (float32) and its own
        # implementation of the Gaussian filter construction.
        score = tf.image.ssim(
            image.cpu().numpy()[None, ..., None],
            reference.cpu().numpy()[None, ..., None],
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
        ).numpy()
        return torch.tensor(score, device=image.device, dtype=image.dtype)

    return tensorflow_ssim
