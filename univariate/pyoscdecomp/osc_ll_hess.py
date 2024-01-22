import numpy as np
import numpy.linalg as LA


def osc_ll_hess(Y, fs: float, param):
    """
    Args:
        Y: ndarray, shape = (J, T)
        fs: float
        param: ndarray, shape = (1, (2*J+1)*K + 1)

    Returns:
        hess: ndarray, 
              shape = ((2*J+1)*K + 1, (2*J+1)*K + 1)
        grad: ndarray, shape = ((2*J+1)*K + 1, 1)
        mll: float
    """    
    J = Y.shape[0]
    T = Y.shape[1]
    length_param = param.shape[1]
    K = (length_param - 1) // (3 + 2 * (J - 1))
    a = param[0, 0:K]
    f = param[0, K:2*K]
    sigma2 = param[0, 2*K:3*K]
    c = param[0, 3*K:-1]
    tau2 = param[0, -1]
    theta = 2 * np.pi * f / fs
    F = np.zeros((2 * K, 2 * K))
    Q = np.zeros((2 * K, 2 * K))
    for k in range(K):
        F[2*k:2*k+2, 2*k:2*k+2] = a[k] * np.block([
            [np.cos(theta[k]), -np.sin(theta[k])], 
            [np.sin(theta[k]),  np.cos(theta[k])]])
        Q[2*k:2*k+2, 2*k:2*k+2] = sigma2[k] * np.eye(2)

    H = np.zeros((J, 2 * K))
    H[0, np.arange(0, 2 * K, 2)] = 1
    kk = 0
    for k in range(K):
        for j in range(1, J):
            H[j, 2*k:2*k+2] = c[kk:kk+2]
            kk += 2

    R = tau2 * np.eye(J)
    grad_F = np.zeros((2 * K, 2 * K, length_param))
    grad_Q = np.zeros((2 * K, 2 * K, length_param))
    grad_H = np.zeros((J, 2 * K, length_param))
    hess_F = np.zeros((2 * K, 2 * K, length_param, length_param))
    kk = 0
    for k in range(K):
        grad_F[2*k:2*k+2, 2*k:2*k+2, k] = np.block([
            [np.cos(theta[k]), -np.sin(theta[k])], 
            [np.sin(theta[k]), np.cos(theta[k])]])
        grad_F[2*k:2*k+2, 2*k:2*k+2, K + k] = a[k] * np.block([
            [-np.sin(theta[k]), -np.cos(theta[k])],
            [np.cos(theta[k]), -np.sin(theta[k])]])
        grad_Q[2*k:2*k+2, 2*k:2*k+2, 2 * K + k] = np.eye(2)
        for j in range(1, J):
            grad_H[j, 2 * k, 3 * K + kk] = 1
            grad_H[j, 2 * k + 1, 3 * K + kk + 1] = 1
            kk += 2
        hess_F[2*k:2*k+2, 2*k:2*k+2, k, K + k] = np.block([
            [-np.sin(theta[k]), -np.cos(theta[k])], 
            [np.cos(theta[k]), -np.sin(theta[k])]])
        hess_F[2*k:2*k+2, 2*k:2*k+2, K + k, k] = np.block([
            [-np.sin(theta[k]), -np.cos(theta[k])], 
            [np.cos(theta[k]), -np.sin(theta[k])]])
        hess_F[2*k:2*k+2, 2*k:2*k+2, K + k, K + k] = a[k] * np.block([
            [-np.cos(theta[k]), np.sin(theta[k])], 
            [-np.sin(theta[k]), -np.cos(theta[k])]])
    
    grad_R = np.zeros((J, J, length_param))
    grad_R[:, :, -1] = np.eye(J)

    x_pred1 = np.zeros((2 * K, 1))
    V_pred1 = np.zeros((2 * K, 2 * K))
    x_filt = np.zeros((2 * K, 1))
    V_filt = np.zeros((2 * K, 2 * K))
    grad_x_pred1 = np.zeros((2 * K, length_param))
    grad_V_pred1 = np.zeros((2 * K, 2 * K, param.shape[1]))
    grad_x_filt = np.zeros((2 * K, length_param))
    grad_V_filt = np.zeros((2 * K, 2 * K, length_param))
    hess_x_pred1 = np.zeros((2 * K, length_param, length_param))
    hess_V_pred1 = np.zeros((2 * K, 2 * K, length_param, length_param))
    hess_x_filt = np.zeros((2 * K, length_param, length_param))
    hess_V_filt = np.zeros((2 * K, 2 * K, length_param, length_param))
    for k in range(K):
        V_pred1[2*k:2*k+2, 2*k:2*k+2] = sigma2[k] / (1 - a[k]**2) * np.eye(2)
        grad_V_pred1[2*k:2*k+2, 2*k:2*k+2, k] = (2 * a[k] * sigma2[k] 
                                                 / (1 - a[k]**2)**2 
                                                 * np.eye(2))
        grad_V_pred1[2*k:2*k+2, 2*k:2*k+2, 2 * K + k] = (1 / (1 - a[k]**2) 
                                                         * np.eye(2))
        hess_V_pred1[2*k:2*k+2, 2*k:2*k+2, k, k] = (2 * (3 * a[k]**2 + 1) 
                                                    * sigma2[k] 
                                                    / (1-a[k]**2)**3 
                                                    * np.eye(2))
        hess_V_pred1[2*k:2*k+2, 2*k:2*k+2, k, 2 * K + k] = (2 * a[k] 
                                                            / (1 - a[k]**2)**2 
                                                            * np.eye(2))
        hess_V_pred1[2*k:2*k+2, 2*k:2*k+2, 2 * K + k, k] = (2 * a[k] 
                                                            / (1-a[k]**2)**2 
                                                            * np.eye(2))

    ll = - J * T/2 * np.log(2 * np.pi)
    grad = np.zeros((length_param, 1))
    hess = np.zeros((length_param, length_param))
    for t in range(T):
        err = Y[:, t:t+1] - H @ x_pred1
        err_cov = H @ V_pred1 @ H.T + R
        inv_err_cov = LA.inv(err_cov)
        ll -= np.log(LA.det(err_cov)) / 2 + err.T @ inv_err_cov @ err / 2
        grad_err = np.zeros((J, length_param))
        for i in range(length_param):
            grad_err[:, i:i+1] = (
                - H @ grad_x_pred1[:, i:i+1] - grad_H[:, :, i] @ x_pred1)
        grad_err_cov = np.zeros((J, J, length_param))
        for i in range(length_param):
            grad_err_cov[:, :, i] = (
                H @ grad_V_pred1[:, :, i] @ H.T 
                + grad_H[:, :, i] @ V_pred1 @ H.T 
                + H @ V_pred1 @ grad_H[:, :, i].T + grad_R[:, :, i])
        hess_err = np.zeros((J, length_param, length_param))
        hess_err_cov = np.zeros((J, J, length_param, length_param))
        for i in range(length_param):
            for j in range(i, length_param):
                hess_err[:, i, j] = (
                    - grad_H[:, :, i] @ grad_x_pred1[:, j] 
                    - grad_H[:, :, j] @ grad_x_pred1[:, i] 
                    - H @ hess_x_pred1[:, i, j])
                hess_err[:, j, i] = hess_err[:, i, j]
                hess_err_cov[:, :, i, j] = (
                    grad_H[:, :, i] @ grad_V_pred1[:, :, j] @ H.T 
                    + grad_H[:, :, j] @ grad_V_pred1[:, :, i] @ H.T 
                    + H @ hess_V_pred1[:, :, i, j] @ H.T 
                    + H @ grad_V_pred1[:, :, i] @ grad_H[:, :, j].T 
                    + H @ grad_V_pred1[:, :, j] @ grad_H[:, :, i].T 
                    + grad_H[:, :, i] @ V_pred1 @ grad_H[:, :, j].T 
                    + grad_H[:, :, j] @ V_pred1 @ grad_H[:, :, i].T)

        for i in range(length_param):
            grad[i, 0] -= (
                (np.trace(inv_err_cov @ grad_err_cov[:, :, i]) 
                 + 2 * err.T @ inv_err_cov @ grad_err[:, i] 
                 - err.T @ inv_err_cov 
                 @ grad_err_cov[:, :, i] @ inv_err_cov @ err) / 2).item()

        for i in range(length_param):
            for j in range(i, length_param):
                hess[i, j] -= (
                    (-np.trace(
                        inv_err_cov @ grad_err_cov[:, :, i] 
                        @ inv_err_cov @ grad_err_cov[:, :, j])
                     + np.trace(inv_err_cov 
                     @ hess_err_cov[:, :, i, j])) / 2).item()
                hess[i, j] -= (
                    (2*grad_err[:, i].T @ inv_err_cov @ grad_err[:, j] 
                     - 2 * err.T @ inv_err_cov
                     @ grad_err_cov[:, :, j] 
                     @ inv_err_cov @ grad_err[:, i] 
                     + 2 * err.T @ inv_err_cov @ hess_err[:, i, j])
                    / 2).item()
                hess[i, j] -= (
                    (-2 * grad_err[:, j].T @ inv_err_cov 
                     @ grad_err_cov[:, :, i] @ inv_err_cov @ err 
                     + 2 * err.T @ inv_err_cov @ grad_err_cov[:, :, j] 
                     @ inv_err_cov @ grad_err_cov[:, :, i] 
                     @ inv_err_cov @ err 
                     - err.T @ inv_err_cov 
                     @ hess_err_cov[:, :, i, j] @ inv_err_cov @ err)
                    / 2).item()
                hess[j, i] = hess[i, j]

        if t == (T - 1):
            break

        Kg = V_pred1 @ H.T @ inv_err_cov
        grad_Kg = np.zeros((2 * K, J, length_param))
        for i in range(length_param):
            grad_Kg[:, :, i] = (
                (grad_V_pred1[:, :, i] @ H.T 
                 + V_pred1 @ grad_H[:, :, i].T) @ inv_err_cov 
                - V_pred1 @ H.T @ inv_err_cov 
                @ grad_err_cov[:, :, i] @ inv_err_cov)

        hess_Kg = np.zeros((2 * K, J, length_param, length_param))
        for i in range(length_param):
            for j in range(i, length_param):
                hess_Kg[:, :, i, j] = (
                    (hess_V_pred1[:, :, i, j] @ H.T 
                     + grad_V_pred1[:, :, i] @ grad_H[:, :, j].T 
                     + grad_V_pred1[:, :, j] @ grad_H[:, :, i].T) 
                    @ inv_err_cov)
                hess_Kg[:, :, i, j] -= (
                    (grad_V_pred1[:, :, i] @ H.T 
                     + V_pred1 @ grad_H[:, :, i].T)
                    @ inv_err_cov @ grad_err_cov[:, :, j] @ inv_err_cov)
                hess_Kg[:, :, i, j] -= (
                    (grad_V_pred1[:, :, j] @ H.T 
                     + V_pred1 @ grad_H[:, :, j].T) 
                    @ inv_err_cov @ grad_err_cov[:, :, i] @ inv_err_cov)
                hess_Kg[:, :, i, j] += (V_pred1 @ H.T @ (
                    inv_err_cov @ grad_err_cov[:, :, j] 
                    @ inv_err_cov @ grad_err_cov[:, :, i] @ inv_err_cov 
                    - inv_err_cov @ hess_err_cov[:, :, i, j] @ inv_err_cov 
                    + inv_err_cov @ grad_err_cov[:, :, i] 
                    @ inv_err_cov @ grad_err_cov[:, :, j] @ inv_err_cov))
                hess_Kg[:, :, j, i] = hess_Kg[:, :, i, j]

        x_filt = x_pred1 + Kg @ err
        V_filt = (np.eye(2 * K) - Kg @ H) @ V_pred1
        for i in range(length_param):
            grad_x_filt[:, i:i+1] = (
                grad_x_pred1[:, i:i+1] 
                + Kg @ grad_err[:, i:i+1] 
                + grad_Kg[:, :, i] @ err)
            grad_V_filt[:, :, i] = (
                grad_V_pred1[:, :, i] 
                - grad_Kg[:, :, i] @ H @ V_pred1 
                - Kg @ grad_H[:, :, i] @ V_pred1 
                - Kg @ H @ grad_V_pred1[:, :, i])

        for i in range(length_param):
            for j in range(i, length_param):
                hess_x_filt[:, i, j] = (
                    hess_x_pred1[:, i, j] 
                    + grad_Kg[:, :, i] @ grad_err[:, j] 
                    + grad_Kg[:, :, j] @ grad_err[:, i] 
                    + Kg @ hess_err[:, i, j] 
                    + (hess_Kg[:, :, i, j] @ err).squeeze())
                hess_x_filt[:, j, i] = hess_x_filt[:, i, j]
                hess_V_filt[:, :, i, j] = (
                    hess_V_pred1[:, :, i, j] 
                    - hess_Kg[:, :, i, j] @ H @ V_pred1 
                    - grad_Kg[:, :, i] @ grad_H[:, :, j] @ V_pred1 
                    - grad_Kg[:, :, i] @ H @ grad_V_pred1[:, :, j] 
                    - grad_Kg[:, :, j] @ grad_H[:, :, i] @ V_pred1 
                    - Kg @ grad_H[:, :, i] @ grad_V_pred1[:, :, j] 
                    - Kg @ grad_H[:, :, i] @ grad_V_pred1[:, :, j] 
                    - grad_Kg[:, :, j] @ H @ grad_V_pred1[:, :, i] 
                    - Kg @ grad_H[:, :, j] @ grad_V_pred1[:, :, i] 
                    - Kg @ H @ hess_V_pred1[:, :, i, j])
                hess_V_filt[:, :, j, i] = hess_V_filt[:, :, i, j]

        x_pred1 = F @ x_filt
        V_pred1 = F @ V_filt @ F.T + Q
        for i in range(length_param):
            grad_x_pred1[:, i:i+1] = (
                F @ grad_x_filt[:, i:i+1] 
                + grad_F[:, :, i] @ x_filt)
            grad_V_pred1[:, :, i] = (
                F @ grad_V_filt[:, :, i] @ F.T 
                + grad_F[:, :, i] @ V_filt @ F.T 
                + F @ V_filt @ grad_F[:, :, i].T 
                + grad_Q[:, :, i])

        for i in range(length_param):
            for j in range(i, length_param):
                hess_x_pred1[:, i, j] = (
                    grad_F[:, :, i] @ grad_x_filt[:, j] 
                    + grad_F[:, :, j] @ grad_x_filt[:, i] 
                    + F @ hess_x_filt[:, i, j] 
                    + (hess_F[:, :, i, j] @ x_filt).squeeze())
                hess_x_pred1[:, j, i] = hess_x_pred1[:, i, j]
                hess_V_pred1[:, :, i, j] = (
                    grad_F[:, :, i] @ grad_V_filt[:, :, j] @ F.T 
                    + grad_F[:, :, j] @ grad_V_filt[:, :, i] @ F.T 
                    + F @ hess_V_filt[:, :, i, j] @ F.T 
                    + F @ grad_V_filt[:, :, i] @ grad_F[:, :, j].T 
                    + F @ grad_V_filt[:, :, j] @ grad_F[:, :, i].T 
                    + hess_F[:, :, i, j] @ V_pred1 @ F.T 
                    + grad_F[:, :, i] @ V_pred1 @ grad_F[:, :, j].T 
                    + grad_F[:, :, j] @ V_pred1 @ grad_F[:, :, i].T 
                    + F @ V_pred1 @ hess_F[:, :, i, j].T)
                hess_V_pred1[:, :, j, i] = hess_V_pred1[:, :, i, j]

    grad[K:2*K] *= 2 * np.pi / fs
    hess[K:2*K, :] *= 2 * np.pi / fs
    hess[:, K:2*K] *= 2 * np.pi / fs
    
    mll = -ll.item()
    grad = -grad
    hess = -hess

    return hess, grad, mll