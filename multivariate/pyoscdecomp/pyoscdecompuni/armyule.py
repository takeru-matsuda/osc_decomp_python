import numpy as np
import numpy.linalg as LA
from scipy.optimize import fminbound
from .ARwithnoise_ll import ARwithnoise_ll


def armyule(y, ARdeg: int):
    """
    Args:
        y: ndarray, shape = (1, T)
        ARdeg: int

    Returns:
        A: ndarray, shape = (1, ARdeg + 1)
        E: float
        R: float
    """    
    T = y.shape[1]
    acov = np.zeros((ARdeg + 1, 1))
    
    acov[:, 0] = np.array(list(
        y[0, 0:T-k] @ y[0, k:T].T / T for k in range(ARdeg + 1)))

    C = np.zeros((ARdeg, ARdeg))
    for k in range(ARdeg):
        C[k, 0:k+1] = acov[np.arange(k, -1, -1), 0]
        C[k, k+1:ARdeg] = acov[1:ARdeg-k, 0]

    c = acov[1:ARdeg+1, 0:1]
    
    [eigs, eigenvectors] = LA.eig(np.block([
                        [C, np.flipud(c)],
                        [np.fliplr(c.T), acov[0:1, 0:1]]
                        ]))
                
    Rmin = 0
    Rmax = np.min(eigs)
    R = fminbound(
        (lambda R, y: ARwithnoise_ll(
            y, 
            np.block([[LA.solve(-(C - R * np.eye(ARdeg)), c)],
                      [np.array(
                        [np.log(acov[0, 0] - R 
                         - LA.solve((C - R * np.eye(ARdeg)), c).T 
                         @ acov[1:ARdeg+1, 0]) 
                         - np.log(R)])]]).T)[0]), 
        Rmin, Rmax, args=(y,),
        xtol=1e-4)
    A = LA.solve(C - R * np.eye(ARdeg), c)
    E = acov[0, 0] - R - (A.T @ acov[1:ARdeg+1, 0]).item()
    A = np.block([1,  -A.T])
    return A, E, R


