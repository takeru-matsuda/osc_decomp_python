import numpy as np
import numpy.linalg as LA

def whittle_multi_ll(X, y, logbeta):
    """
    Args:
        X: ndarray, shape = (J, J, [T/2] + 1, K + 1)
        y: ndarray, shape = (J, [T/2] + 1)
        logbeta: ndarray, shape = (K+1, 1)

    Returns:
        mll: float
    """    
    d = X.shape[0]
    K = X.shape[3]
    beta = np.exp(logbeta)
    mll = 0.0

    for i in range(y.shape[1]):
        F = np.zeros((d, d)).astype(complex)
        for k in range(K):
            F += 2 * np.pi * X[:, :, i, k] * beta[k, 0]
        mll += (
            np.log(LA.det(F)) 
            + np.conjugate(y[:, i:i+1]).T @ LA.solve(F, y[:, i:i+1]))
        mll = mll.item().real
        
    return mll
