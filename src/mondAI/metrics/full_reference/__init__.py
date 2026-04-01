from .fsim import FSIM
from .haarpsi import HaarPSI, HaarPSI_MED
from .mae import MAE
from .ssim import SSIM
from .vifp import VIFP

FULL_REFERENCE_METRICS = [MAE, SSIM, HaarPSI, HaarPSI_MED, FSIM, VIFP]
