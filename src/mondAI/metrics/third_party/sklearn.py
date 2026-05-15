from collections.abc import Callable

import torch


def get_sklearn_mae() -> Callable[..., torch.Tensor]:
    from sklearn.metrics import mean_absolute_error

    def sklearn_mae(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Flatten the tensors and convert to numpy arrays for sklearn
        image_np = image.cpu().numpy().flatten()
        reference_np = reference.cpu().numpy().flatten()
        mae_value = mean_absolute_error(reference_np, image_np)
        return torch.tensor(mae_value, device=image.device)

    return sklearn_mae


def get_sklearn_mse() -> Callable[..., torch.Tensor]:
    from sklearn.metrics import mean_squared_error as mse_sklearn

    def sklearn_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Flatten the tensors and convert to numpy arrays for sklearn
        image_np = image.cpu().numpy().flatten()
        reference_np = reference.cpu().numpy().flatten()
        mse_value = mse_sklearn(reference_np, image_np)
        return torch.tensor(mse_value, device=image.device)

    return sklearn_mse


def get_sklearn_rmse() -> Callable[..., torch.Tensor]:
    from sklearn.metrics import root_mean_squared_error as rmse_sklearn

    def sklearn_rmse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Flatten the tensors and convert to numpy arrays for sklearn
        image_np = image.cpu().numpy().flatten()
        reference_np = reference.cpu().numpy().flatten()
        rmse_value = rmse_sklearn(reference_np, image_np)
        return torch.tensor(rmse_value, device=image.device)

    return sklearn_rmse
