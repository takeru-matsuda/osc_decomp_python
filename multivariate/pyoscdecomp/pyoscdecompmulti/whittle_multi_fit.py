import numpy as np
from scipy.optimize import minimize
from .whittle_multi_ll import whittle_multi_ll

def whittle_multi_fit(X, y):
    """
    Args:
        X: ndarray, shape = (J, J, [T/2] + 1, K + 1)
        y: ndarray, shape = (J, [T/2] + 1)

    Returns:
        beta: ndarray, shape = (K+1, 1)
    """    
    sizeX = X.shape[3]
    res = minimize(
        (lambda b, X, y:
            whittle_multi_ll(
                X, y, b.reshape((sizeX, 1)))), 
        np.zeros(sizeX),
        method='BFGS',
        args=(X, y),
        tol=1e-6,
        options={'disp': False}
    )
    beta = np.exp(res.x).reshape((sizeX, 1))
    return beta


