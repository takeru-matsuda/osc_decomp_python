import numpy as np
import numpy.linalg as LA


def check_eigenvector(C, X, E):
    """
    ベクトル X と値の組 E が、行列のリスト C から定まる
    多項式固有値問題
    (C[0] + E C[1] + E^2 C[2] + ... + E^p C[p]) X = 0
    を解くかを確認する関数です。
    Args:
        C: list  of ndarray (shape = (N, N))
            Coefficient matrices
        X: ndarray, shape = (N, N*p)
            Eigen vectors
        E: ndarray, shape = (N*p)
            Eigen values
    Returns:
        res: ndarray, shape = (N)
            residual
    """    
    n = len(C[0][0])
    p = len(C[0][:]) - 1

    res = list()
    for ieig in range(n*p):
        lam = E[ieig]
        M = C[0][0].astype(complex)
        for i in range(1, p+1):
            M += lam * C[0][i].astype(complex)
            lam *= E[ieig]
        res.append(M @ X[:, ieig])

    res = np.array(res)

    return res













