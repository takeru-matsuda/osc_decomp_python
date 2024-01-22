import numpy as np


def whittle_uni_ll(X, y, beta):    
    """
    Args:
        X: ndarray, shape = ([T/2] + 1, K + 1)
        y: ndarray, shape = ([T/2] + 1, 1)
        beta: ndarray, shape = (K + 1, 1)

    Returns:
        mll: float
    """    
    Xbeta = X @ beta
    mll = np.sum(np.log(Xbeta) + y/Xbeta)
    return mll
