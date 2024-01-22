import numpy as np
from scipy.optimize import minimize
from .whittle_uni_ll import whittle_uni_ll


def whittle_uni_fit(X, y):
    """
    Args:
        X: ndarray, shape = ([T/2] + 1, K + 1)
        y: ndarray, shape = ([T/2] + 1, 1)

    Returns:
        beta: ndarray, shape = (K + 1, 1)
    """
    sizeX = X.shape[1]
    res = minimize(
        (lambda b, X, y:
            whittle_uni_ll(
                X, y, np.exp(b).reshape((sizeX, 1)))), 
        np.zeros(sizeX),
        method='BFGS',
        args=(X, y),
        tol=1e-6
    )

    beta = np.exp(res.x).reshape((sizeX, 1))

    return beta
