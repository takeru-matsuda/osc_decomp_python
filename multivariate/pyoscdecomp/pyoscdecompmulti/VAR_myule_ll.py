import numpy as np
import numpy.linalg as LA
from .polyeig_VAR import polyeig_VAR
from .VARwithnoise_ll import VARwithnoise_ll

def VAR_myule_ll(Y, r: float, C, c):
    """
    Args:
        Y: ndarray, shape = (J, T)
        r: float
        C: ndarray, shape = (J*ARdeg, J*ARdeg)
        c: ndarray, shape = (J*ARdeg, J)

    Returns:
        mll: float
        A: ndarray, shape = (J, J*ARdeg)
        E: ndarray, shape = (J, J)
    """    
    d = Y.shape[0]
    ARdeg = C.shape[0] // d
    B = LA.solve(C - r * np.eye(d*ARdeg), c)
    A = np.zeros((d, d*ARdeg))
    for k in range(ARdeg):
        A[:, k*d:(k+1)*d] = B[k*d:(k+1)*d, :].T

    E = C[0:d, 0:d] - r * np.eye(d)
    for k in range(ARdeg):
        E -= A[:, k*d:(k+1)*d] @ c[k*d:(k+1)*d, :]

    V, z = polyeig_VAR(A)
    if ((np.max(np.abs(z)) < 1) and (np.min(LA.eig(E)[0]) > 0)):
        mll = VARwithnoise_ll(Y, A, E, r*np.eye(d))
    else:
        mll = np.inf

    return mll, A, E
