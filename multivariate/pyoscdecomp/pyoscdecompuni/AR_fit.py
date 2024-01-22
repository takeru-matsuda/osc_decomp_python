import numpy as np
from spectrum import aryule
from .AR_ll import AR_ll
from .ARwithnoise_ll import ARwithnoise_ll
from .armyule import armyule


def AR_fit(y, MAX_AR: int):
    """
    Args:
        y: ndarray, shape = (1, T)
        MAX_AR: int

    Returns:
        ARwithnoise_param: ndarray, shape = (MAX_AR, MAX_AR + 2) 
        ARwithnoise_AIC: ndarray, shape = (1, MAX_AR) 
        AR_param: ndarray, shape = (MAX_AR, MAX_AR + 1) 
        AR_AIC: ndarray, shape = (1, MAX_AR)
    """    
    AR_AIC = np.zeros((1, MAX_AR))
    AR_param = np.zeros((MAX_AR, MAX_AR + 1))
    ARwithnoise_AIC = np.zeros((1, MAX_AR))
    ARwithnoise_param = np.zeros((MAX_AR, MAX_AR + 2))

    for ARdeg in range(1, MAX_AR + 1):
        [A0, E, k] = aryule(y[0], ARdeg)
        A = np.hstack([1, A0])
        a = np.zeros((ARdeg, ARdeg))
        a[:, ARdeg-1] = -A[1:ARdeg+1]

        for m in range(ARdeg, 1, -1):
            for i in range(1, m):
                a[i-1, m-2] = (
                    (a[i - 1, m - 1] + a[m - 1, m - 1] * a[m - i - 1, m - 1]) 
                    / (1 - a[m - 1, m - 1]**2))
        c = np.zeros((1, ARdeg))

        c[0, :] = np.array(list(a[m, m] for m in range(ARdeg)))
        
        AR_AIC[0, ARdeg - 1] = (
            2 * AR_ll(y, np.log(1 + c) - np.log(1 - c))[0] + 2 * (ARdeg + 1))        
        AR_param[ARdeg - 1, 0:ARdeg+1] = np.block([A[1:ARdeg+1],  E])
        [A, E, R] = armyule(y, ARdeg)

        param = np.block([[A[0, 1:ARdeg+1], np.log(E) - np.log(R)]])
        [mll, R] = ARwithnoise_ll(y, param)
        ARwithnoise_AIC[0, ARdeg - 1] = 2 * mll + 2 * (ARdeg + 2)
        ARwithnoise_param[ARdeg - 1, 0:ARdeg+2] = np.block(
            [param[0, 0:ARdeg], 
             np.exp(param[0, ARdeg:ARdeg+1]) * R, 
             R * np.ones(1)])

    return ARwithnoise_param, ARwithnoise_AIC, AR_param, AR_AIC
