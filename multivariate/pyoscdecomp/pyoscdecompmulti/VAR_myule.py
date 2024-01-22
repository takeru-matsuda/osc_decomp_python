import numpy as np
import numpy.linalg as LA
from scipy.optimize import fminbound
from .VAR_myule_ll import VAR_myule_ll

def VAR_myule(Y, ARdeg: int):
    """
    Args:
        Y: ndarray, shape = (J, T)
        ARdeg: int

    Returns:
        mll: float
        A: ndarray, shape = (J, J*(ARdeg + 1))
        E: ndarray, shape = (J, J)
    """    
    d = Y.shape[0]
    T = Y.shape[1]
    acov = np.zeros((d, d, ARdeg+1))
    Y = Y - np.mean(Y, axis=1).reshape(d, 1) @ np.ones((1, T))
    for k in range(ARdeg+1):
        acov[:, :, k] = Y[:, 0:T-k] @ Y[:, k:T].T / T
    C = np.zeros((d*ARdeg, d*ARdeg))
    for k in range(ARdeg):
        for j in range(k+1):
            C[k*d:(k+1)*d, j*d:(j+1)*d] = acov[:, :, k-j]
        for j in range(k+1, ARdeg):
            C[k*d:(k+1)*d, j*d:(j+1)*d] = acov[:, :, j-k].T
    c = np.zeros((d*ARdeg, d))
    for k in range(ARdeg):
        c[k*d:(k+1)*d, :] = acov[:, :, k+1]

    r = fminbound(
        (lambda r, Y, C, c: VAR_myule_ll(Y, r, C, c)[0]),
        0.0, np.min(LA.eig(C)[0]),
        args=(Y, C, c),
        xtol=1e-4
    )
    mll, A, E = VAR_myule_ll(Y, r, C, c)
    A = np.block([np.eye(d), -A])

    return A, E, r, mll
