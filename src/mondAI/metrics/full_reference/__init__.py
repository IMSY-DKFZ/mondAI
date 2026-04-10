from .fsim import FSIM
from .haarpsi import HaarPSI, HaarPSI_MED
from .mae import MAE
from .mse import MSE
from .vifp import VIFP

FULL_REFERENCE_METRICS = [MAE, MSE, HaarPSI, HaarPSI_MED, FSIM, VIFP]
