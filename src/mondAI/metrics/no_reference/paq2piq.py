import os
from typing import Callable

import torch
import torchvision as tv
from torch.hub import get_dir, load_state_dict_from_url

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.no_reference.base import NoReferenceMetric
from mondAI.utils.internal_format import _get_torch_device

logger = get_logger()


class PaQ2PiQ(NoReferenceMetric):
    """From Patches to Pictures (PaQ-2-PiQ) measures the perceptual quality of images
    blindly based on features extracted from a pretrained model. Here the RoIPool Model
    of the original publication is implemented, not the Feedback model. The model
    architecture is based on a ResNet18 backbone which was pretrained with ImageNet and
    then finetuned on the FLIVE dataset. The model weights are frozen and downloaded
    upon first usage (and cached for future use), internet is required for first usage.
    Besides computing a global score, the metric also computes local scores for
    different patches of the image, which can be used to identify regions of the image
    that are of particularly high or low quality. The metric expects RGB images as
    input, with pixel values in the range [0, 1]. The output is a global score in the
    range [0, 100], where higher is better.

    Implementation adapted from the original author's implementation under
    https://github.com/baidut/paq2piq/ commit: 48c91e844e9f7a768f6ddcfc744dddfdd1160fea, License: MIT.

    Original publication:
    Ying, Zhenqiang, et al. "From patches to pictures (PaQ-2-PiQ): Mapping the perceptual space of picture quality."
    Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

    """

    @property
    def name(self) -> str:
        return "From Patches to Pictures"

    @property
    def abbreviation(self) -> str:
        return "PaQ-2-PiQ"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)  # RGB expected

    def __init__(
        self,
        model_weights_url: str = "https://github.com/baidut/PaQ-2-PiQ/releases/download/v1.0/RoIPoolModel-fit.10.bs.120.pth",
    ) -> None:
        """Initialize PaQ-2-PiQ metric."""
        super().__init__()
        self.model_weights_url = model_weights_url

        # check that URL is valid
        if not self.model_weights_url.startswith("http"):
            raise ValueError(f"model_weights_url must be a valid URL, got {self.model_weights_url}")

        # download model weights if not already downloaded and store in cache directory, and load weights
        model_weights_path = os.path.join(get_dir(), "mondAI")
        try:
            logger.info(
                f"Loading model weights from {model_weights_path}. "
                f"If not present, downloading from {self.model_weights_url}..."
            )
            model_state_dict = load_state_dict_from_url(
                self.model_weights_url,
                model_dir=model_weights_path,
                map_location=lambda storage, loc: storage,
                progress=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download and load model weights from {self.model_weights_url}. "
                f"Please check the URL and your internet connection. Error: {e}"
            ) from e

        self.model = RoIPoolModel()
        self.model.load_state_dict(model_state_dict["model"])
        self.model.to(_get_torch_device())
        self.model.eval()

    def _compute(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the metric for the given image. Images must be RGB and between [0,
        1], returns a score between [0, 100], where higher is better.

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :return: The computed metric score.
        :rtype: torch.Tensor

        """

        # check input value range
        if torch.any(image < 0) or torch.any(image > 1):
            raise ValueError(
                "Input image contains pixel values outside the range [0, 1], "
                f"got min {image.min()} and max {image.max()}."
            )

        # check for RGB channels
        channel_dim = self.expected_dimensions.index(Dimension.CHANNEL)
        if image.shape[channel_dim] != 3:
            raise ValueError(f"Image has {image.shape[channel_dim]} channels. PaQ-2-PiQ expects 3 (RGB) channels")

        # actual model prediction
        image = image.float().unsqueeze(0)  # convert to float and add batch dimension
        self.model.input_block_rois(block_size=(20, 20), img_size=image.shape[-2:], device=image.device)
        output = self.model(image)
        global_score = output[0, 0]
        # local_scores = output[0, 1:].reshape(20, 20)
        return global_score  # ,local_scores

    def _other_implementations(self) -> dict[str, Callable[..., torch.Tensor]]:
        """Return a dictionary of other implementations of the metric. This will be
        used when compare_implementations is True to compute the metric using different
        libraries or implementations for comparison.

        :return: A dictionary where the keys are the names of the libraries or implementations, and the values are
        callables that compute the metric using those implementations.
        :rtype: dict[str, callable[..., torch.Tensor]]

        """

        return {}

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        return (
            f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()} "
            f"with pretrained weights from {self.model_weights_url} "
        )


class AdaptiveConcatPool2d(torch.nn.Module):  # type: ignore[misc]
    """Layer that concatenates the results of adaptive average pooling and adaptive max
    pooling.

    Adapted from the original author's implementation under
    https://github.com/baidut/paq2piq/blob/master/paq2piq/model.py,
    commit: 0136db158b1ca7b0bc8a33021541d05e8b7d90db, License: MIT

    """

    def __init__(self, size: tuple[int, int] | None = None) -> None:
        super().__init__()
        size = size or (1, 1)
        self.average_pool = torch.nn.AdaptiveAvgPool2d(size)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.max_pool(x), self.average_pool(x)], 1)


class RoIPoolModel(torch.nn.Module):  # type: ignore[misc]
    """Model architecture for the RoIPool Model.

    Adapted from the original author's implementation under
    https://github.com/baidut/paq2piq/blob/master/paq2piq/inference_model.py,
    commit: 8d7d8cc39fa4e397b5eb2626eecbd99f3a72636a, License: MIT

    """

    rois = None

    def __init__(self) -> None:
        super().__init__()

        model = tv.models.resnet18()
        cut = -2
        spatial_scale = 1 / 32

        self.model_type = self.__class__.__name__
        self.body = torch.nn.Sequential(*list(model.children())[:cut])
        self.head = torch.nn.Sequential(
            AdaptiveConcatPool2d(),
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Dropout(p=0.25, inplace=False),
            torch.nn.Linear(in_features=1024, out_features=512, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=512, out_features=1, bias=True),
        )
        self.roi_pool = tv.ops.RoIPool((2, 2), spatial_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.body(x)
        batch_size = x.size(0)

        if self.rois is not None:
            rois_data = self.rois.view(-1, 4)
            n_output = int(rois_data.size(0) / batch_size)
            indices = (
                torch.arange(batch_size, device=x.device, dtype=torch.float32).repeat_interleave(n_output).unsqueeze(1)
            )
            indexed_rois = torch.cat((indices, rois_data), 1)
            feats = self.roi_pool(feats, indexed_rois)
        preds = self.head(feats)
        return preds.view(batch_size, -1)

    def input_block_rois(
        self, block_size: tuple[int, int] = (20, 20), img_size: tuple[int, int] = (1, 1), device: torch.device = None
    ) -> None:
        ys = torch.linspace(0, img_size[0], steps=block_size[0] + 1, device=device)
        xs = torch.linspace(0, img_size[1], steps=block_size[1] + 1, device=device)

        grid_y0, grid_x0 = torch.meshgrid(ys[:-1], xs[:-1], indexing="ij")
        grid_y1, grid_x1 = torch.meshgrid(ys[1:], xs[1:], indexing="ij")

        blockwise_rois = torch.stack([grid_x0, grid_y0, grid_x1, grid_y1], dim=-1).reshape(-1, 4)
        global_rois = torch.tensor([[0, 0, img_size[1], img_size[0]]], device=device)
        self.rois = torch.cat((global_rois, blockwise_rois))
