import numpy as np
from .mypolyeig import mypolyeig

def polyeig_VAR(A):
    """
    Args:
        A: ndarray, shape = (J, J * ARdeg)
    Returns:
        V: ndarray, shape = (J, J * ARdeg)
        z: ndarray, shape = (J * ARdeg)
    """    
    d = A.shape[0]    
    ARdeg = A.shape[1] // d
    C = list([[] * (ARdeg + 1)])
    for k in range(ARdeg):
        C[0].insert(k, -A[:, (ARdeg-k-1)*d:(ARdeg-k)*d])

    C[0].insert(ARdeg, np.eye(d))

    [V, z, _] = mypolyeig(C)

    return V, z

