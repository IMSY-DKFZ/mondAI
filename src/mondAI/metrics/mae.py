import torch


class MAE:
    def __init__(self) -> None:
        pass

    def compute(self, y_true: torch.tensor, y_pred: torch.tensor) -> float:
        """
        Compute Mean Absolute Error (MAE) between true and predicted values.

        Parameters:
        y_true (list or array-like): True values.
        y_pred (list or array-like): Predicted values.

        Returns:
        float: Mean Absolute Error.
        """
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        mae: float = torch.mean(torch.abs(y_true_tensor - y_pred_tensor)).item()
        return mae
