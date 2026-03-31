from .fsim import FSIM
from .haarpsi import HaarPSI, HaarPSI_MED
from .mae import MAE
from .vifp import VIFP

FULL_REFERENCE_METRICS = [MAE, HaarPSI, HaarPSI_MED, FSIM, VIFP]
