from collections import namedtuple
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal, no_type_check

import torch
import torchvision

from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.deepinv import get_deepinv_lpips
from mondAI.metrics.third_party.piq import get_piq_lpips
from mondAI.metrics.third_party.piqa import get_piqa_lpips
from mondAI.metrics.third_party.torchmetrics import get_torchmetrics_lpips


class LPIPS(FullReferenceMetric):
    r"""Learned Perceptual Image Patch Similarity (LPIPS).

    LPIPS is a perceptual similarity metric that compares deep features extracted from a pre-trained
    convolutional neural network (CNN) to assess the perceptual similarity between two images.
    It is designed to align with human judgments of image quality.
    LPIPS computes a weighted cosine distance between the deep features of the input image and the reference image
    across multiple layers of the CNN with the following formula:

    .. math::
        \text{LPIPS}(x, x_0)=\sum_{l} \frac{1}{H_l W_l} \sum_{h,w} ||w_l \cdot (\hat{y}_{hw}^l - \hat{y}_{0hw}^l)||_2^2

    where:
        - \\(x\\) is the input image,
        - \\(x_0\\) is the reference image,
        - \\(l\\) indexes the layers of the CNN,
        - \\(H_l\\) and \\(W_l\\) are the height and width of the feature maps at layer \\(l\\),
        - \\(w_l\\) are the learned weights for layer \\(l\\),
        - \\(\hat{y}_{hw}^l\\) and \\(\hat{y}_{0hw}^l\\) are the normalized deep features at
            spatial location \\((h,w)\\) for the input and reference images, respectively.

    LPIPS expects RGB images with a value range of [0, 1], which are internally standardized to ImageNet statistics
    in range [-1, 1]. This implementation uses the version 0.1 weights provided by the original authors.

    Implementation taken from Richard Zhang:
    https://github.com/richzhang/PerceptualSimilarity, commit: 082bb24f84c091ea94de2867d34c4544f68e0963
    license: BSD 2-Clause License, with slight adaptations for weight loading to fit the metric class structure.

    Original publication:
    Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018).
    The unreasonable effectiveness of deep features as a perceptual metric.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 586-595).

    """

    @property
    def name(self) -> str:
        return "Learned Perceptual Image Patch Similarity"

    @property
    def abbreviation(self) -> str:
        return "LPIPS"

    @property
    def higher_is_better(self) -> bool:
        return False

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.CHANNEL, Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(
        self,
        network_architecture: Literal["vgg", "alex", "squeeze"] = "alex",
    ) -> None:
        """

        :param network_architecture: The architecture of the pre-trained CNN to use for feature extraction.
            Choices are "vgg", "alex", or "squeeze", default is "alex".
        :type network_architecture: str

        """
        super().__init__()
        self.network_architecture = network_architecture

        if network_architecture not in ["vgg", "alex", "squeeze"]:
            raise ValueError(
                f"Invalid network architecture '{network_architecture}'. Valid options are 'vgg', 'alex', or 'squeeze'."
            )

        self.model = LPIPS_Module(
            pretrained=True,
            net=self.network_architecture,
            version="0.1",
            lpips=True,
            spatial=False,
            pnet_rand=False,
            pnet_tune=False,
            use_dropout=True,
            model_path=None,
            eval_mode=True,
            verbose=False,
        )

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute the metric between image and reference.

        :param image: The input image for which the metric is being computed.
        :type image: torch.Tensor
        :param reference: The reference image to compare against.
        :type reference: torch.Tensor
        :return: The computed metric score.
        :rtype: torch.Tensor

        """
        self._input_checks(image, reference)

        # normalize from [0,1] to [-1,1]
        image = 2 * image - 1
        reference = 2 * reference - 1

        self.model.to(image.device).double()
        score = self.model(reference.unsqueeze(0), image.unsqueeze(0))
        return score.squeeze()

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        """Override this method in subclasses to register other implementations of the
        metric for comparison. Use the `_register_implementation` helper method to add
        implementations to the internal dictionary. These implementations will be used
        when compare_implementations is True.

        Store reference implementations in the `third_party` submodule of the metrics module,
        and import them here to register them for comparison.

        """
        self._register_implementation(
            implementations, "torchmetrics", get_torchmetrics_lpips(self.network_architecture)
        )
        self._register_implementation(implementations, "piq", get_piq_lpips())  # piq only has vgg16 version
        self._register_implementation(implementations, "piqa", get_piqa_lpips(self.network_architecture))
        self._register_implementation(implementations, "deepinv", get_deepinv_lpips(self.network_architecture))

    def __str__(self) -> str:
        """Full text representation of the metric.

        :return: A string representation of the metric including its name,
            abbreviation, and an arrow indicating whether higher values are better.
        :rtype: str

        """
        return (
            f"{self.name} ({self.abbreviation}) {self._arrow_indicating_optimum()} with "
            f"network_architecture={self.network_architecture}"
        )

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to LPIPS, such as checking for valid pixel
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
            LPIPS.

        """
        # Input checks specific to LPIPS
        if torch.any(image < 0) or torch.any(image > 1):
            raise ValueError("Input image contains pixel values outside the range [0, 1].")

        if torch.any(reference < 0) or torch.any(reference > 1):
            raise ValueError("Reference image contains pixel values outside the range [0, 1].")

        channel_dim = self.expected_dimensions.index(Dimension.CHANNEL)
        if image.shape[channel_dim] != 3:
            raise ValueError(f"Images have {image.shape[channel_dim]} channels, but LPIPS expects 3 (RGB) channels.")


"""
Below code is taken from the original authors
"""


@no_type_check
def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


@no_type_check
def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    return torch.nn.Upsample(size=out_HW, mode="bilinear", align_corners=False)(in_tens)


# Learned perceptual metric
@no_type_check
class LPIPS_Module(torch.nn.Module):  # type: ignore
    @no_type_check
    def __init__(
        self,
        pretrained=True,
        net="alex",
        version="0.1",
        lpips=True,
        spatial=False,
        pnet_rand=False,
        pnet_tune=False,
        use_dropout=True,
        model_path=None,
        eval_mode=True,
        verbose=True,
    ):
        """Initializes a perceptual loss torch.nn.Module.

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained a normalization bug; corresponds to old arxiv v1 (https://arxiv.org/abs/1801.03924v1)
        model_path : 'str'
            [None] is default and loads the pretrained weights from paper https://arxiv.org/abs/1801.03924v1

        The following parameters should only be changed if training the network

        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] keep base/trunk frozen
            [True] tune the base/trunk network
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers

        """

        super().__init__()
        if verbose:
            print(
                f"Setting up [{'LPIPS' if lpips else 'baseline'} perceptual loss: trunk [{net}], v[{version}], "
                f"spatial [{'on' if spatial else 'off'}]"
            )

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == "squeeze":  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = torch.nn.ModuleList(self.lins)

            if pretrained:
                if model_path is None:
                    model_path = str(
                        PurePosixPath(Path.cwd() / f"src/mondAI/metrics/full_reference/lpips_weights/{net}.pth")
                    )

                if verbose:
                    print(f"Loading model from: {model_path}")
                self.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)

        if eval_mode:
            self.eval()

    @no_type_check
    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == "0.1" else (in0, in1)
        )
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        @no_type_check
        def normalize_tensor(in_feat, eps=1e-10):
            norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
            return in_feat / (norm_factor + eps)

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if self.spatial:
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for layer in range(self.L):
            val += res[layer]

        if retPerLayer:
            return (val, res)
        else:
            return val


@no_type_check
class ScalingLayer(torch.nn.Module):  # type: ignore
    @no_type_check
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    @no_type_check
    def forward(self, inp):
        return (inp - self.shift) / self.scale


@no_type_check
class NetLinLayer(torch.nn.Module):  # type: ignore
    """A single linear layer which does a 1x1 conv."""

    @no_type_check
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()

        layers = (
            [
                torch.nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            torch.nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = torch.nn.Sequential(*layers)

    @no_type_check
    def forward(self, x):
        return self.model(x)


@no_type_check
class squeezenet(torch.nn.Module):  # type: ignore
    @no_type_check
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        pretrained_features = torchvision.models.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    @no_type_check
    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        SqueezeOutputs = namedtuple("SqueezeOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7"])
        return SqueezeOutputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)


@no_type_check
class alexnet(torch.nn.Module):  # type: ignore
    @no_type_check
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        alexnet_pretrained_features = torchvision.models.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    @no_type_check
    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        AlexnetOutputs = namedtuple("AlexnetOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"])
        return AlexnetOutputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


@no_type_check
class vgg16(torch.nn.Module):  # type: ignore
    @no_type_check
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    @no_type_check
    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        VggOutputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return VggOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
