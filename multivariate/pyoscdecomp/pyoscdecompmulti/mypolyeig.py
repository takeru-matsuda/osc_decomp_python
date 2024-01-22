import sys
import numpy as np
import numpy.linalg as LA
from scipy.linalg import eig
from typing import Optional
# slight modification of MATLAB polyeig.

def mypolyeig(
    C, 
    if_calculate_eigenvector: Optional[bool] = True, 
    if_calculate_cond: Optional[bool] = False):
    """
    Args:
        C: list [1, ndarray((n, n))*(p+1)]
        if_calculate_eigenvector: bool. Defaults to True.
        if_calculate_cond: bool. Defaults to False.
    Returns:
        X: ndarray, shape = (n, n*p)
        E: ndarray, shape = (n*p)
        s: ndarray, shape = (n*p, 1)
    """    
    eps = sys.float_info.epsilon
    n = len(C[0][0])
    p = len(C[0][:]) - 1

    if (if_calculate_cond and (p < 1)):
        message = (
            "mypolyeig: tooFewInputs. "
            + "Must provide at least two matrices.")
        raise ValueError(message)
    X = np.array([]).astype(complex)
    E = np.array([]).astype(complex)
    s = None
    A = np.eye(n*p)
    A[0:n, 0:n] = C[0][0]

    if (p == 0):
        B = np.eye(n)
        p = 1
    else:
        B = np.diag(np.ones(n*(p-1)), -n)
        j  = np.arange(n)
        for k in range(p):
            B[0:n, j] = -C[0][k+1]
            j += n
    # Use the QZ algorithm on the big matrix pair (A,B).
    if (if_calculate_eigenvector):
        res_eig = eig(A, B, left=False, right=True)
        X = np.array(res_eig[1]).astype(complex)
        E = np.array(res_eig[0]).astype(complex)
    else:
        res_eig = eig(A, B, left=False, right=True)
        E = np.array(res_eig[0]).astype(complex)
        return X, E, s
    if (p >= 2):
        # For each eigenvalue, extract the eigenvector from whichever portion
        # of the big eigenvector matrix X gives the smallest normalized residual.
        V = np.zeros((n, p)).astype(complex)
        for j in range(p*n):
            V[:, :] = X[:, j].reshape((n, p), order='F')
            R = C[0][p]
            if not np.isinf(E[j]):
                for k in range(p-1, -1, -1):
                    R = C[0][k] + E[j] * R

            R = R @ V
            res = np.sum(np.abs(R), axis=0) / np.sum(np.abs(V), axis=0)  # Normalized residuals.
            ind = np.argmin(res)
            X[0:n, j] = V[:, ind] / LA.norm(V[:, ind])  # Eigenvector with unit 2-norm.
        X = X[0:n, :]
    if (if_calculate_eigenvector and (not if_calculate_cond)):
        return X, E, s
    if if_calculate_eigenvector and if_calculate_cond:
        # Construct matrix Y whose rows are conjugates of left eigenvectors.
        rcond_p = 1 / LA.cond(C[0][p])
        rcond_0 = 1 / LA.cond(C[0][0])
        if (max(rcond_p, rcond_0) <= eps):
            message = (
                'mypolyeig:nonSingularCoeffMatrix' 
                + 'Either the leading or '
                + 'the trailing coefficient matrix must be nonsingular.')
            raise ValueError(message)

        if (rcond_p >= rcond_0):
            V = C[0][p]
            E1 = E
        else:
            V = C[0][0]
            E1 = 1 / E
        Y = X
        for i in range(p-1):
            Y = (np.block(
                [[Y], 
                 [Y[-n:, :] @ np.diag(E1)]]))
        B = np.zeros((p*n, n))
        B[-n:, 0:n] = np.eye(n)
        Y = LA.solve(V.T, LA.solve(Y, B).T).T
        for i in range(n*p):
            Y[i, :] = Y[i, :] / LA.norm(Y[i, :]) # Normalize left eigenvectors.
        # Reconstruct alpha-beta representation of eigenvalues: E(i) = a(i)/b(i).
        a = E
        b = np.ones(len(a))
        k = np.isinf(a)
        a[k] = 1 
        b[k] = 0
    
        nu = np.zeros((p+1, 1))
        for i in range(p+1):
            nu[i, 0] = LA.norm(C[0][i], 'fro')
        s = np.zeros((n*p, 1))

        # Compute condition numbers.
        for j in range(p*n):
            ab = (a[j]**range(p)) * (b[j]**range(p-1, -1, -1))            
            Da = ab[0] * C[0][1]
            Db = p * ab[0] * C[0][0]
            for k in range(1, p):
                Da += (k+1) * ab[k] * C[0][k+1]
                Db += (p-k) * ab[k] * C[0][k]
            nab = LA.norm(
                    (a[j]**range(p+1)) 
                    * (b[j]**range(p, -1, -1)) * nu.T)
            s[j, 0] = nab / np.abs(
                Y[j, :] 
                @ (np.conjugate(b[j])*Da - np.conjugate(a[j])*Db) @ X[:, j])

        return X, E, s
