from typing import Optional
from .osc_decomp_uni import osc_decomp_uni


def osc_decomp(
    Y, fs, 
    MAX_OSC: Optional[int] = None, MAX_AR: Optional[int] = None, 
    algorithm: Optional[str] = None, grad: Optional[bool] = None):
    """
    Args:
        Y: ndarray, shape = (J, T)
        fs: float
        MAX_OSC: int, Defaults to None.
        MAX_AR: int, Defaults to None.
        algorithm: str, Defaults to None.
        grad: bool Defaults to None.

    Returns:
        osc_param: ndarray, shape = (MAX_OSC, 3*MAX_OSC + 1)
        osc_AIC: ndarray, shape = (1, MAX_OSC)
        osc_mean: ndarray, shape = (2*MAX_OSC, 2*MAX_OSC, T, MAX_OSC)
        osc_phase: ndarray, shape = (MAX_OSC, T, MAX_OSC)
    """               
    if (MAX_OSC is None):
        MAX_OSC = 5

    if (MAX_AR is None):
        MAX_AR = max(20, 2 * MAX_OSC)

    if (Y.shape[0] == 1):
        [
            osc_param, 
            osc_AIC, 
            osc_mean, 
            osc_cov,
            osc_phase] = osc_decomp_uni(
                Y, fs, 
                MAX_OSC=MAX_OSC, 
                MAX_AR=MAX_AR, 
                algorithm=algorithm, 
                grad=grad)
    else:
        raise ValueError('osc_decom_multi is not implemented yet.')

    return osc_param, osc_AIC, osc_mean, osc_cov, osc_phase
    





