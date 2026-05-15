from .cwssim import CWSSIM
from .dists import DISTS
from .dss import DSS
from .fsim import FSIM
from .gmsd import GMSD
from .haarpsi import HaarPSI, HaarPSI_MED
from .iwssim import IWSSIM
from .lpips import LPIPS
from .mae import MAE
from .mdsi import MDSI
from .mse import MSE
from .msssim import MSSSIM
from .psnr import PSNR
from .ssim import SSIM
from .vifp import VIFP
from .vsi import VSI

FULL_REFERENCE_METRICS = [
    MAE,
    MSE,
    SSIM,
    MSSSIM,
    IWSSIM,
    CWSSIM,
    HaarPSI,
    HaarPSI_MED,
    FSIM,
    VIFP,
    PSNR,
    DSS,
    GMSD,
    MDSI,
    VSI,
    DISTS,
    LPIPS,
]
