import numpy as np
import numpy.linalg as LA

# Relaxed version of A^-1 x
def relaxed_solve(A, x):
    try:
        ret = LA.solve(A, x)
        return ret
    except LA.LinAlgError:
        ret = LA.lstsq(A, x, rcond=1e-8)
        return ret[0]

# Relaxed version of A^-1
def relaxed_inv(A):
    try:
        ret = LA.inv(A)
        return ret
    except LA.LinAlgError:
        ret = LA.pinv(A, rcond=1e-8)
        return ret


#% profile likelihood (observation noise variance reduced by Kitagawa method)
def osc_multi_prof_ll(Y, param, init_theta, return_grad: bool):
    """
    Args:
        Y: ndarray, shape = (J, T)
        param: ndarray, shape = ((2*J+1)* K)
        init_theta: ndarray, shape = (1, K)
        return_grad: bool

    Returns:
        mll: float
        grad: ndarray, shape = ((2*J+1)*K, 1)
        tau2hat: float
    """    
    J = Y.shape[0]
    T = Y.shape[1]
    K = len(param) // (2*J + 1)
    a = (np.tanh(param[0:K]) + 1) / 2
    theta = init_theta.squeeze() + np.tanh(param[K:2*K]) * np.pi
    sigma2 = np.exp(param[2*K:3*K])
    c = param[3*K:len(param)]
    F = np.zeros((2*K, 2*K))
    Q = np.zeros((2*K, 2*K))
    for k in range(K):
        F[2*k:2*k+2, 2*k:2*k+2] = (a[k] * np.block([
            [np.cos(theta[k]), -np.sin(theta[k])],
            [np.sin(theta[k]), np.cos(theta[k])]]))
        Q[2*k:2*k+2, 2*k:2*k+2] = sigma2[k] * np.eye(2)

    H = np.zeros((J, 2*K))
    H[0, np.arange(0, 2 * K, 2)] = 1
    kk = 0
    for k in range(K):
        for j in range(1, J):
            H[j, 2*k:2*k+2] = c[kk:kk+2]
            kk += 2

    R = np.eye(J)
    x_pred1 = np.zeros((2*K, T))
    x_filt = np.zeros((2*K, T))
    V_pred1 = np.zeros((2*K, 2*K, T))
    V_filt = np.zeros((2*K, 2*K, T))
    x_pred1[:, 0:1] = np.zeros((2*K, 1))
    for k in range(K):
        """original
        V_pred1[2*k:2*k+2, 2*k:2*k+2, 0] = (
            sigma2[k] / (1 - a[k]**2) * np.eye(2))
        """
        # modified for numerical stability
        denominator_factor = (
            (np.exp(2*param[k]) + 1)/2
            + 1 / 2 /  (1 + a[k]))
        V_pred1[2*k:2*k+2, 2*k:2*k+2, 0] = (
            sigma2[k] * denominator_factor * np.eye(2))

    
    for t in range(T-1):
        x_filt[:, t:t+1] = (
            x_pred1[:, t:t+1] 
            + V_pred1[:, :, t] @ H.T @ relaxed_solve(
                H @ V_pred1[:, :, t] @ H.T + R, 
                Y[:, t:t+1] - H @ x_pred1[:, t:t+1]))
        V_filt[:, :, t] = (
            V_pred1[:, :, t] 
            - V_pred1[:, :, t] @ H.T @ relaxed_solve(
                H @ V_pred1[:, :, t] @ H.T + R, H) @ V_pred1[:, :, t])
        x_pred1[:, t+1:t+2] = F @ x_filt[:, t:t+1]
        V_pred1[:, :, t+1] = F @ V_filt[:, :, t] @ F.T + Q

    tau2hat = 0.0
    for t in range(T):
        tau2hat += (
            (Y[:, t:t+1] - H @ x_pred1[:, t:t+1]).T 
            @ relaxed_solve(
                H @ V_pred1[:, :, t] @ H.T + R, 
                Y[:, t:t+1] - H @ x_pred1[:, t:t+1]) 
            / J / T).item()

    ll = - J * T * np.log(tau2hat) / 2 - J * T/2 - J * T/2 * np.log(2 * np.pi)
    for t in range(T):
        ll -= np.log(LA.det(H @ V_pred1[:, :, t] @ H.T + R)) / 2
    mll = -ll
    
    if return_grad == False:
        grad = 0.0
        return mll, grad, tau2hat

    grad_F = np.zeros((2*K, 2*K, (2*J+1)*K))
    grad_Q = np.zeros((2*K, 2*K, (2*J+1)*K))
    grad_H = np.zeros((J, 2*K, (2*J+1)*K))
    kk = 0
    for k in range(K):
        grad_F[2*k:2*k+2, 2*k:2*k+2, k] = (
            1 / 2 / np.cosh(param[k]) / np.cosh(param[k]) * np.block(
                [[np.cos(theta[k]), -np.sin(theta[k])], 
                 [np.sin(theta[k]), np.cos(theta[k])]]))
        grad_F[2*k:2*k+2, 2*k:2*k+2, K+k] = (
            np.pi / np.cosh(param[K + k]) /  np.cosh(param[K + k])  * a[k] * np.block(
                [[-np.sin(theta[k]), -np.cos(theta[k])], 
                 [np.cos(theta[k]), -np.sin(theta[k])]]))
        grad_Q[2*k:2*k+2, 2*k:2*k+2, 2*K+k] = sigma2[k] * np.eye(2)
        for j in range(1, J):
            grad_H[j, 2*k, 3*K+kk] = 1
            grad_H[j, 2*k+1, 3*K+kk+1] = 1
            kk += 2

    grad_x_pred1 = np.zeros((2*K, (2*J+1)*K))
    grad_V_pred1 = np.zeros((2*K, 2*K, (2*J+1)*K))
    for k in range(K):
        """original
        grad_V_pred1[2*k:2*k+2, 2*k:2*k+2, k] = (
            a[k] / np.cosh(param[k])**2 * sigma2[k] 
            / (1 - a[k]**2)**2 * np.eye(2))
        grad_V_pred1[2*k:2*k+2, 2*k:2*k+2, 2*K+k] = (
            sigma2[k] / (1 - a[k]**2) * np.eye(2))
        """            
        # modified for numerical stability        
        denominator_factor = (
            (np.exp(2*param[k]) + 1)/2
            + 1 / 2 /  (1 + a[k]))
        grad_V_pred1[2*k:2*k+2, 2*k:2*k+2, k] = (
            a[k] / np.cosh(param[k]) / np.cosh(param[k]) * sigma2[k] 
            * denominator_factor * denominator_factor *  np.eye(2))
        grad_V_pred1[2*k:2*k+2, 2*k:2*k+2, 2*K+k] = (
            sigma2[k] * denominator_factor * np.eye(2))

    grad_x_filt = np.zeros((2*K, (2*J+1)*K))
    grad_V_filt = np.zeros((2*K, 2*K, (2*J+1)*K))
    grad_tau2hat = np.zeros(((2*J+1)*K, 1))
    grad = np.zeros(((2*J+1)*K, 1))
    for t in range(T):
        err = Y[:, t:t+1] - H @ x_pred1[:, t:t+1]
        err_cov = H @ V_pred1[:, :, t] @ H.T + R
        inv_err_cov = relaxed_inv(err_cov)
        grad_err = np.zeros((J, (2*J+1)*K))
        for i in range((2*J+1)*K):
            grad_err[:, i:i+1] = (
                - H @ grad_x_pred1[:, i:i+1] 
                - grad_H[:, :, i] @ x_pred1[:, t:t+1])
        grad_err_cov = np.zeros((J, J, (2*J+1)*K))
        for i in range((2*J+1)*K):
            grad_err_cov[:, :, i] = (
                H @ grad_V_pred1[:, :, i] @ H.T 
                + grad_H[:, :, i] @ V_pred1[:, :, t] @ H.T 
                + H @ V_pred1[:, :, t] @ grad_H[:, :, i].T)
        for i in range((2*J+1)*K):
            grad[i, 0] -= (np.trace(inv_err_cov @ grad_err_cov[:, :, i])) / 2
            grad_tau2hat[i, 0] -= (
                2 * (Y[:, t:t+1] - H @ x_pred1[:, t:t+1]).T 
                @ inv_err_cov 
                @ (H @ grad_x_pred1[:, i:i+1] 
                   + grad_H[:, :, i] @ x_pred1[:, t:t+1]) / J / T).item()
            grad_tau2hat[i, 0] -= (
                (Y[:, t:t+1] - H @ x_pred1[:, t:t+1]).T 
                @ inv_err_cov 
                @ grad_err_cov[:, :, i] 
                @ inv_err_cov 
                @ (Y[:, t:t+1] - H @ x_pred1[:, t:t+1]) / J / T).item()

        if (t == T-1):
            break

        Kg = V_pred1[:, :, t] @ H.T @ inv_err_cov
        grad_Kg = np.zeros((2*K, J, (2*J+1)*K))
        for i in range((2*J+1)*K):
            grad_Kg[:, :, i] = (
                (grad_V_pred1[:, :, i] @ H.T 
                 + V_pred1[:, :, t] @ grad_H[:, :, i].T) @ inv_err_cov 
                 - V_pred1[:, :, t] @ H.T @ inv_err_cov 
                 @ grad_err_cov[:, :, i] @ inv_err_cov)

        for i in range((2*J+1)*K):
            grad_x_filt[:, i:i+1] = (
                grad_x_pred1[:, i:i+1] 
                + Kg @ grad_err[:, i:i+1] 
                + grad_Kg[:, :, i] @ err)
            grad_V_filt[:, :, i] = (
                grad_V_pred1[:, :, i] 
                - grad_Kg[:, :, i] @ H @ V_pred1[:, :, t] 
                - Kg @ grad_H[:, :, i] @ V_pred1[:, :, t] 
                - Kg @ H @ grad_V_pred1[:, :, i])
        for i in range((2*J+1)*K):
            grad_x_pred1[:, i:i+1] = (
                F @ grad_x_filt[:, i:i+1] + grad_F[:, :, i] @ x_filt[:, t:t+1])
            grad_V_pred1[:, :, i] = (
                F @ grad_V_filt[:, :, i] @ F.T 
                + grad_F[:, :, i] @ V_filt[:, :, t] @ F.T 
                + F @ V_filt[:, :, t] @ grad_F[:, :, i].T 
                + grad_Q[:, :, i])
    grad -= J * T * grad_tau2hat / tau2hat / 2
    grad = -grad

    return mll, grad, tau2hat

