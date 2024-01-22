import numpy as np
from .VAR_myule import VAR_myule

def VAR_fit(Y, MAX_VAR: int):
    """
    Args:
        Y: ndarray, shape = (J, T)
        MAX_VAR: int

    Returns:
        VARwithnoise_A: ndarray, shape = (J, J*MAX_VAR, MAX_VAR)
        VARwithnoise_E: ndarray, shape = (J, J, MAX_VAR)
        VARwithnoise_r: ndarray, shape = (1, MAX_VAR)
        VARwithnoise_AIC: ndarray, shape = (1, MAX_VAR)
    """    
    J = Y.shape[0]
    VARwithnoise_AIC = np.zeros((1, MAX_VAR))
    VARwithnoise_A = np.zeros((J, J*MAX_VAR, MAX_VAR))
    VARwithnoise_E = np.zeros((J, J, MAX_VAR))
    VARwithnoise_r = np.zeros((1, MAX_VAR))
    for ARdeg in range(1, MAX_VAR+1):
        A, E, r, mll = VAR_myule(Y, ARdeg)
        VARwithnoise_A[:, 0:J*ARdeg, ARdeg-1] = -A[:, J:(ARdeg+1)*J]
        VARwithnoise_E[:, :, ARdeg-1] = E
        VARwithnoise_r[0, ARdeg-1] = r
        VARwithnoise_AIC[0, ARdeg-1] = (
            2 * mll + 2 * (J**2 * ARdeg + J*(J+1)/2 + 1))

    return VARwithnoise_A, VARwithnoise_E, VARwithnoise_r, VARwithnoise_AIC
