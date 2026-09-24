# SPDX-FileCopyrightText: 2026 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Callable

import torch

from mondAI.logging import get_logger
from mondAI.metrics.dimension import Dimension
from mondAI.metrics.full_reference.base import FullReferenceMetric
from mondAI.metrics.third_party.medimetrics import get_medimetrics_cwssim
from mondAI.utils.signal_processing import gaussian_filter_kernel

logger = get_logger()


class CWSSIM(FullReferenceMetric):
    r"""Complex-Wavelet Structural Similarity Index Measure (CW-SSIM).

    The Complex-Wavelet Structural Similarity Index Measure (CW-SSIM) compares two
    images in a complex wavelet domain. Unlike pixel-domain similarity measures, it is
    designed to be insensitive to small translations, rotations, and other geometric
    distortions that produce approximately consistent local phase shifts in complex
    wavelet coefficients.

    For local complex coefficients :math:`c_x` and :math:`c_y`, the local CW-SSIM map
    is defined as

    .. math::
        \operatorname{CW\text{-}SSIM}(x, y) =
        \frac{2 \left| \sum_i c_{x,i} c_{y,i}^* \right| + K}
        {\sum_i |c_{x,i}|^2 + \sum_i |c_{y,i}|^2 + K},

    where :math:`c_{y,i}^*` denotes the complex conjugate of :math:`c_{y,i}` and
    :math:`K` is a small stabilizing constant. The final score is obtained by averaging
    weighted local scores across orientations.

    Implementation adapted from the original MATLAB code by Zhou Wang, Mehul Sampat and Alan Bovik
    (https://de.mathworks.com/matlabcentral/fileexchange/43017-complex-wavelet-structural-similarity-index-cw-ssim/files/cwssim_index.m,
     license: BSD 2-Clause)

    Original publication:
    M. P. Sampat, Z. Wang, S. Gupta, A. C. Bovik, and M. K. Markey,
    "Complex Wavelet Structural Similarity: A New Image Similarity Index",
    IEEE Transactions on Image Processing, vol. 18, no. 11, pp. 2385-2401, 2009,
    doi: 10.1109/TIP.2009.2025923.

    """

    @property
    def name(self) -> str:
        return "Complex-Wavelet Structural Similarity Index Measure"

    @property
    def abbreviation(self) -> str:
        return "CW-SSIM"

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def scaling_factor(self) -> float:
        return 255.0

    @property
    def expected_dimensions(self) -> tuple[Dimension, ...]:
        return (Dimension.HEIGHT, Dimension.WIDTH)

    def __init__(
        self,
        levels: int = 6,
        orientations: int = 16,
        guard_boundary: int = 0,
        k: float = 0.0,
    ) -> None:
        """Initialize CW-SSIM.

        :param levels: Number of pyramid levels used for complex steerable pyramid
            decomposition, needs to be positive (default: 6)
        :type levels: int
        :param orientations: Number of orientations used for complex steerable pyramid
            decomposition, needs to be positive (default: 16)
        :type orientations: int
        :param guard_boundary: Size of guard boundary in pixels to exclude from
            computation, needs to be non-negative (default: 0)
        :type guard_boundary: int
        :param k: Stabilizing constant, needs to be non-negative (default: 0.0)
        :type k: float
        :raises ValueError: If levels is not positive, orientations is not positive,
            guard_boundary is negative, or k is negative.

        """
        super().__init__()
        self.levels = levels
        self.orientations = orientations
        self.guard_boundary = guard_boundary
        self.k = k

        if self.levels < 1:
            raise ValueError(f"levels must be at least 1, but got {self.levels}.")
        if self.orientations < 1:
            raise ValueError(f"orientations must be at least 1, but got {self.orientations}.")
        if self.guard_boundary < 0:
            raise ValueError(f"guard_boundary must be non-negative, but got {self.guard_boundary}.")
        if self.k < 0:
            raise ValueError(f"k must be non-negative, but got {self.k}.")

    def _compute(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute CW-SSIM between image and reference."""
        self._input_checks(image, reference)

        cw_image = self._get_pyramid_bands(image, self.levels, self.orientations)
        cw_reference = self._get_pyramid_bands(reference, self.levels, self.orientations)
        s = image.shape[0] / (2 ** (self.levels - 1))
        w = gaussian_filter_kernel(s - 7 + 1 - 2 * self.guard_boundary, s / 4, image.device, image.dtype)
        window_size = 7
        win7 = torch.ones(1, 1, window_size, window_size, device=image.device, dtype=torch.complex128) / float(
            window_size**2
        )

        band_cssims = image.new_zeros(self.orientations)
        for i in range(self.orientations):
            band_image = cw_image[i]
            band_reference = cw_reference[i]
            if self.guard_boundary > 0:
                band_image = band_image[
                    self.guard_boundary : -self.guard_boundary, self.guard_boundary : -self.guard_boundary
                ]
                band_reference = band_reference[
                    self.guard_boundary : -self.guard_boundary, self.guard_boundary : -self.guard_boundary
                ]
            corr = band_image * torch.conj(band_reference)
            corr_band = torch.nn.functional.conv2d(corr[None, None, :, :], win7)
            varr = torch.abs(band_image) ** 2 + torch.abs(band_reference) ** 2
            varr_band = torch.nn.functional.conv2d(varr[None, None, :, :], win7.real)
            cssim_map = (2 * torch.abs(corr_band) + self.k) / (varr_band + self.k)
            band_cssims[i] = (cssim_map * w[None, None, :, :]).sum(dim=[-2, -1]).squeeze()

        return band_cssims.mean()

    def _get_pyramid_bands(self, image: torch.Tensor, height: int, orientations: int) -> dict[int, torch.Tensor]:
        """Construct complex steerable pyramid decomposition of the image and return
        the complex coefficients for the specified height and orientations.

        Implementation adapted from pyrtools package
        (https://github.com/LabForComputationalVision/pyrtools, license: MIT)

        :param image: Input image for which to compute the complex steerable pyramid decomposition, shape (H, W)
        :type image: torch.Tensor
        :param height: Number of pyramid levels to compute, needs to be positive
        :type height: int
        :param orientations: Number of orientations to compute, needs to be positive
        :type orientations: int
        :return: Dictionary mapping orientation index to complex coefficients for the specified height and orientation
        :rtype: dict[int, torch.Tensor]

        """

        max_height = math.floor(math.log2(min(image.shape))) - 2
        if height > max_height:
            raise ValueError(
                f"Requested pyramid height {height} exceeds maximum possible height {max_height} "
                f"for image size {image.shape}."
            )

        coefficients = {}
        twidth = 1
        dims = torch.tensor(image.shape, device=image.device, dtype=image.dtype)
        ctr = torch.ceil((dims + 0.5) / 2).int()

        xramp, yramp = torch.meshgrid(
            torch.linspace(-1, 1, int(dims[1] + 1), dtype=image.dtype, device=image.device)[:-1],
            torch.linspace(-1, 1, int(dims[0] + 1), dtype=image.dtype, device=image.device)[:-1],
            indexing="ij",
        )

        angle = torch.arctan2(yramp, xramp)
        log_rad = torch.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0] - 1, ctr[1] - 1] = log_rad[ctr[0] - 1, ctr[1] - 2]
        log_rad = torch.log2(log_rad)

        def rcosFN(width: float, position: float, values: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
            """Generate the raised cosine function used for constructing the radial
            transition functions in the pyramid decomposition."""
            sz = 256
            X = torch.pi * torch.arange(-sz - 1, 2, device=image.device, dtype=image.dtype) / (2 * sz)
            Y = values[0] + (values[1] - values[0]) * torch.cos(X) ** 2
            Y[0] = Y[1]
            Y[sz + 2] = Y[sz + 1]
            X = position + (2 * width / torch.pi) * (X + torch.pi / 4)
            return X, Y

        def interp1(x: torch.Tensor, y: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
            """Perform 1D linear interpolation of the values in y at the positions in x
            for the query points in query.

            This is used for constructing the radial and angular transition functions in the pyramid decomposition.
            The implementation is similar to the MATLAB `interp1` function which the linear extrapolation behavior

            """
            slope = torch.diff(y) / torch.diff(x)
            offset = y[:-1] - slope * x[:-1]
            indices = torch.searchsorted(x, query, right=False)
            indices = torch.clamp(indices - 1, 0, len(slope) - 1)
            return slope.gather(-1, indices) * query + offset.gather(-1, indices)

        def pointOp(im: torch.Tensor, lut: torch.Tensor, origin: torch.Tensor, increment: torch.Tensor) -> torch.Tensor:
            """Apply a point operation to the image im using the lookup table lut.

            The values in lut are assumed to be equally spaced with spacing increment
            and starting at origin.

            """
            X = origin + increment * torch.arange(0, len(lut), device=im.device, dtype=image.dtype)
            Y = lut
            interp = interp1(X, Y, im.flatten())
            return interp.reshape(im.shape)

        Xrcos, Yrcos = rcosFN(twidth, -twidth / 2.0, (0, 1))
        Yrcos = torch.sqrt(Yrcos)

        YIrcos = torch.sqrt(1.0 - Yrcos**2)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1] - Xrcos[0])

        imdft = torch.fft.fftshift(torch.fft.fft2(image))

        lo0mask = lo0mask.reshape(imdft.shape)
        lodft = imdft * lo0mask

        for i in range(height):
            Xrcos -= 1

            lutsize = 1024
            Xcosn = (
                torch.pi
                * torch.arange(-(2 * lutsize + 1), (lutsize + 2), device=image.device, dtype=image.dtype)
                / lutsize
            )

            const = (
                (2 ** (2 * orientations - 1))
                * math.factorial(orientations - 1) ** 2
                / float(orientations * math.factorial(2 * (orientations - 1)))
            )

            alfa = ((torch.pi + Xcosn) % (2.0 * torch.pi)) - torch.pi
            Ycosn = (2.0 * math.sqrt(const) * torch.cos(Xcosn) ** (orientations - 1)) * (
                torch.abs(alfa) < torch.pi / 2.0
            ).int()

            log_rad_test = torch.reshape(log_rad, (1, log_rad.shape[0] * log_rad.shape[1]))
            himask = pointOp(log_rad_test, Yrcos, Xrcos[0], Xrcos[1] - Xrcos[0])
            himask = himask.reshape((lodft.shape[0], lodft.shape[1]))

            if i == height - 1:
                for b in range(orientations):
                    angle_tmp = torch.reshape(angle, (1, angle.shape[0] * angle.shape[1]))
                    anglemask = pointOp(angle_tmp, Ycosn, Xcosn[0] + torch.pi * b / orientations, Xcosn[1] - Xcosn[0])
                    anglemask = anglemask.reshape((lodft.shape[0], lodft.shape[1]))

                    banddft = (-1j) ** (orientations - 1) * lodft * anglemask * himask
                    band = torch.fft.ifft2(torch.fft.ifftshift(banddft))
                    coefficients[b] = band

            dims = torch.tensor(lodft.shape, device=image.device, dtype=image.dtype)
            ctr = torch.ceil((dims + 0.5) / 2).int()
            lodims = torch.ceil((dims - 0.5) / 2).int()
            loctr = torch.ceil((lodims + 0.5) / 2).int()
            lostart = ctr - loctr
            loend = lostart + lodims

            log_rad = log_rad[lostart[0] : loend[0], lostart[1] : loend[1]]
            angle = angle[lostart[0] : loend[0], lostart[1] : loend[1]]
            lodft = lodft[lostart[0] : loend[0], lostart[1] : loend[1]]
            YIrcos = torch.abs(torch.sqrt(1.0 - Yrcos**2))
            log_rad_tmp = torch.reshape(log_rad, (1, log_rad.shape[0] * log_rad.shape[1]))
            lomask = pointOp(log_rad_tmp, YIrcos, Xrcos[0], Xrcos[1] - Xrcos[0])
            lomask = lomask.reshape((lodft.shape[0], lodft.shape[1]))
            lodft = lodft * lomask

        return coefficients

    def _register_other_implementations(self, implementations: dict[str, Callable[..., torch.Tensor]]) -> None:
        self._register_implementation(implementations, "medimetrics", get_medimetrics_cwssim())

    def __str__(self) -> str:
        """Full text representation of the metric."""
        arrow = self._arrow_indicating_optimum()
        return (
            f"{self.name} ({self.abbreviation}) {arrow} with parameters: "
            f"{self.levels=}, {self.orientations=}, {self.guard_boundary=}, {self.k=}"
        )

    def _input_checks(self, image: torch.Tensor, reference: torch.Tensor) -> None:
        """Perform input checks specific to CW-SSIM."""
        minimum_size = max(7, 2 * self.guard_boundary + 1)
        if image.shape[-2] < minimum_size or image.shape[-1] < minimum_size:
            raise ValueError(
                f"Images have spatial dimensions {image.shape[-2:]} which are too small for CW-SSIM with "
                f"window_size=7 and guard_band={self.guard_boundary}."
            )
