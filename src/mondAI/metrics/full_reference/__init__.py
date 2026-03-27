from .fsim import FSIM
from .haarpsi import HaarPSI, HaarPSI_MED
from .mae import MAE

FULL_REFERENCE_METRICS = [MAE, HaarPSI, HaarPSI_MED, FSIM]
