from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import no_type_check

import numpy as np
import torch
import torchvision

from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.medimetrics import get_medimetrics_dists
from mondAI.metrics.third_party.piq import get_piq_dists
from mondAI.metrics.third_party.torchmetrics import get_torchmetrics_dists
from mondAI.utils.checks import check_rgb, check_value_range


class DISTS(FullReferenceMetric):
    """Deep Image Structure and Texture Similarity (DISTS)

    DISTS extracts deep features from the image and its reference using a pre-trained VGG16 network and computes
    a similarity score based on the structure (means) and texture (variances) of the images. The structure and texture
    scores are weighted by learned parameters alpha and beta, which were trained to align with human judgments.
    The final DISTS score is computed as 1 - (weighted structure similarity + weighted texture similarity),
    so higher scores indicate better perceptual quality.

    DISTS expects RGB images with a value range of [0, 1], which are internally standardized to ImageNet statistics.
    It is symmetric, meaning that the order of the input images does not matter.
    Internally images are downsampled to a maximum of 256x256 pixels if they are larger than that, which is the source
    of score deviations to other implementations.

    Implementation taken from original author Keyan Ding
    (https://github.com/dingkeyan93/DISTS/blob/master/DISTS_pytorch/DISTS_pt.py,
    commit: 1267d8cb626c98706db3697422701c56a85ebf2e, license: MIT)
    with slight adaptations for weight loading and input parsing and to fit the metric class structure

    Original publication:
    Ding, K., Ma, K., Wang, S., & Simoncelli, E. P. (2020).
    Image quality assessment: Unifying structure and texture similarity.
    IEEE transactions on pattern analysis and machine intelligence, 44(5), 2567-2581

    """

    @property
    def name(self) -> str:
        return "Deep Image Structure and Texture Similarity"

    @property
    def abbreviation(self) -> str:
        return "DISTS"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        if self.batched:
            return (Dimension.BATCH, Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)
        else:
            return (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(self, batched: bool = False) -> None:
        """
        :param batched: Whether the images will be provided in batches and scores should be computed batch-wise.
         If False, the metric expects inputs of shape (C, H, W) and will add a batch dimension internally.
         If True, the metric expects inputs of shape (N, C, H, W). Default is False.
        :type batched: bool
        """
        super().__init__()
        self.batched = batched
        self.model = DISTS_Module()

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        self._input_checks(image, reference)

        # downsample if necessary
        if min(image.shape[self.expected_dimensions.index(dim)] for dim in (Dimension.HEIGHT, Dimension.WIDTH)) > 256:
            image = torchvision.transforms.functional.resize(image, 256)
            reference = torchvision.transforms.functional.resize(reference, 256)

        self.model.to(image.device).double()

        if not self.batched:
            image = image.unsqueeze(0)
            reference = reference.unsqueeze(0)

        score = self.model(reference, image)
        return score.squeeze()

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        """Override this method in subclasses to register other implementations of the
        metric for comparison. Use the `_register_implementation` helper method to add
        implementations to the internal dictionary. These implementations will be used
        when compare_implementations is True.

        Store reference implementations in the `third_party` submodule of the metrics module,
        and import them here to register them for comparison.

        """
        self._register_implementation(implementations, "piq", get_piq_dists(batched=self.batched))
        self._register_implementation(implementations, "torchmetrics", get_torchmetrics_dists(batched=self.batched))
        self._register_implementation(
            implementations, "medimetrics", get_medimetrics_dists()
        )  # only supports non-batched inputs

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        return f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()}, batched={self.batched}"

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to DISTS, such as checking for valid pixel
        value ranges and dimensions.

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :raises ValueError: If the input image contains pixel values outside the range
            [0, 1].
        :raises ValueError: If the reference image contains pixel values outside the
            range [0, 1].
        :raises ValueError: If the images do not have 3 channels, which is required for
            DISTS.

        """
        # Input checks specific to DISTS
        check_value_range(image, 0, 1)
        check_value_range(reference, 0, 1, reference=True)

        check_rgb(self.expected_dimensions, image, self.abbreviation)


@no_type_check
class L2pooling(torch.nn.Module):  # type: ignore
    @no_type_check
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer("filter", g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    @no_type_check
    def forward(self, input):
        input = input**2
        out = torch.nn.functional.conv2d(
            input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1]
        )
        return (out + 1e-12).sqrt()


class DISTS_Module(torch.nn.Module):  # type: ignore
    @no_type_check
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        ).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter("alpha", torch.nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter("beta", torch.nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        weights = torch.load(str(PurePosixPath(Path.cwd() / "src/mondAI/metrics/full_reference/dists_weights.pt")))
        self.alpha.data = weights["alpha"]
        self.beta.data = weights["beta"]

    @no_type_check
    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    @no_type_check
    def forward(self, x, y):
        with torch.no_grad():
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        return 1 - (dist1 + dist2).squeeze()
