from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric

logger = get_logger()


class MSE(FullReferenceMetric):
    r"""Mean Squared Error (MSE).

    The Mean Squared Error (MSE) is a measure of errors between paired images
    expressed as the average squared difference between the given and reference pixel
    values. It is defined mathematically as:

    .. math::
        \operatorname {MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2

    where:

    * \\(y_i\\) is the reference pixel value,
    * \\(\\hat{y}_i \\) is the given pixel value,
    * \\(N\\) is the number of pixels per image.

    """

    @property
    def name(self) -> str:
        return "Mean Squared Error"

    @property
    def abbreviation(self) -> str:
        return "MSE"

    @property
    def higher_is_better(self) -> bool:
        return False

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self) -> None:
        """Initialize MSE metric."""
        super().__init__()

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the Mean Squared Error between image and reference."""
        return torch.mean((image - reference) ** 2)

    def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
        """Return a dictionary of other implementations of the MSE metric."""

        implementations = {}

        ### skimage ###
        try:
            from skimage.metrics import mean_squared_error as mse_skimage

            def skimage_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # Flatten the tensors and convert to numpy arrays for scikit-image
                image_np = image.cpu().numpy().flatten()
                reference_np = reference.cpu().numpy().flatten()
                mse_value = mse_skimage(reference_np, image_np)
                return torch.tensor(mse_value, device=image.device)

            implementations["skimage"] = skimage_mse

        except ImportError:
            logger.warning("scikit-image is not available, skipping scikit-image implementation of MSE.")

        ### sklearn ###
        try:
            from sklearn.metrics import mean_squared_error as mse_sklearn

            def sklearn_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                # Flatten the tensors and convert to numpy arrays for sklearn
                image_np = image.cpu().numpy().flatten()
                reference_np = reference.cpu().numpy().flatten()
                mse_value = mse_sklearn(reference_np, image_np)
                return torch.tensor(mse_value, device=image.device)

            implementations["sklearn"] = sklearn_mse

        except ImportError:
            logger.warning("sklearn is not available, skipping sklearn implementation of MSE.")

        ### torchmetrics ###
        try:
            from torchmetrics.functional.regression import mean_squared_error as mse_torchmetrics

            def torchmetrics_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                return mse_torchmetrics(image.contiguous(), reference.contiguous())

            implementations["torchmetrics"] = torchmetrics_mse

        except ImportError:
            logger.warning("torchmetrics is not available, skipping torchmetrics implementation of MSE.")

        ### tensorflow ###
        try:
            import tensorflow as tf

            def tensorflow_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                image_np = image.detach().cpu().numpy()
                reference_np = reference.detach().cpu().numpy()
                mse_metric = tf.keras.metrics.MeanSquaredError()
                mse_metric.update_state(reference_np, image_np)
                mse_value = mse_metric.result().numpy()
                return torch.as_tensor(mse_value, device=image.device)

            implementations["tensorflow"] = tensorflow_mse

        except ImportError:
            logger.warning("tensorflow is not available, skipping tensorflow implementation of MSE.")

        ### monai ###
        try:
            from monai.metrics import MSEMetric

            def monai_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                mse_metric = MSEMetric()
                mse_value = mse_metric(image.unsqueeze(0), reference.unsqueeze(0))
                return torch.as_tensor(mse_value, device=image.device)

            implementations["monai"] = monai_mse

        except ImportError:
            logger.warning("monai is not available, skipping monai implementation of MSE.")

        ### medimetrics ###
        try:
            from mondAI.metrics.third_party.medimetrics.mse import MSE as MSEMedimetrics

            def medimetric_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                medimetrics_metric = MSEMedimetrics()
                mse_value = medimetrics_metric.compute(image.numpy(force=True), reference.numpy(force=True))
                return torch.as_tensor(mse_value, device=image.device)

            implementations["medimetrics"] = medimetric_mse

        except ImportError:
            logger.warning("medimetrics is not available, skipping medimetrics implementation of MSE.")

        ### deepinv ###
        try:
            from deepinv.loss.metric import MSE as MSEDeepInv

            def deepinv_mse(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                mse_metric = MSEDeepInv()
                mse_value = mse_metric(image.unsqueeze(0).unsqueeze(0), reference.unsqueeze(0).unsqueeze(0))
                return torch.as_tensor(mse_value, device=image.device)

            implementations["deepinv"] = deepinv_mse

        except ImportError:
            logger.warning("deepinv is not available, skipping deepinv implementation of MSE.")

        return implementations

    def __str__(self) -> str:
        """Full text representation of the MSE metric."""
        return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}"
