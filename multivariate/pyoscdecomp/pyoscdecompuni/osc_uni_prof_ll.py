import numpy as np
import numpy.linalg as LA

# profile likelihood (observation noise variance reduced by Kitagawa method)
def osc_uni_prof_ll(y, param, init_theta, return_grad):
    """
    Args:
        y: ndarray, shape = (1, T)
        param: ndarray, shape = (3 * K)
               This parameter is 1-D array to be compatible with 
               scipy.optimize.minimize.
        init_theta: ndarray, shape = (1, K)
        return_grad:bool

    Returns:
        mll: float
        grad: ndarray, shape = (3*K, 1)
        tau2hat: float
    """    
    T = y.shape[1]
    K = param.shape[0] // 3
    a = (np.tanh(param[0:K]) + 1) / 2
    theta = init_theta.squeeze() + np.tanh(param[K:2*K]) * np.pi
    sigma2 = np.exp(param[2*K:3*K])
    F = np.zeros((2 * K, 2 * K))
    Q = np.zeros((2 * K, 2 * K))

    for k in range(K):
        F[2*k:2*k+2, 2*k:2*k+2] = (a[k] * np.block([
            [np.cos(theta[k]), -np.sin(theta[k])],
            [np.sin(theta[k]), np.cos(theta[k])]]))
        Q[2*k:2*k+2, 2*k:2*k+2] = sigma2[k] * np.eye(2)

    H = np.zeros((1, 2 * K))
    H[0, np.arange(0, 2 * K, 2)] = 1
    R = 1

    x_pred1 = np.zeros((2 * K, T))
    x_filt = np.zeros((2 * K, T))
    V_pred1 = np.zeros((2 * K, 2 * K, T))
    V_filt = np.zeros((2 * K, 2 * K, T))
    x_pred1[:, 0:1] = np.zeros((2 * K, 1))

    for k in range(K):
        V_pred1[2*k:2*k+2, 2*k:2*k+2, 0] = (sigma2[k] 
                                            / (1 - a[k]**2) * np.eye(2))

    for t in range(T - 1):
        x_filt[:, t] = x_pred1[:, t] + (V_pred1[:, :, t] @ H.T 
                                        @ LA.solve( 
                                        H @ V_pred1[:, :, t] @ H.T + R,
                                        y[0, t] - H @ x_pred1[:, t]))
        V_filt[:, :, t] = ((np.eye(2 * K) 
                            - (V_pred1[:, :, t] @ (H.T @ H)) 
                            / (H@V_pred1[:, :, t]@H.T + R)) 
                           @ V_pred1[:, :, t])
        x_pred1[:, t+1:t+2] = F @ x_filt[:, t:t+1]
        V_pred1[:, :, t + 1] = F @ V_filt[:, :, t] @ F.T + Q    

    tau2hat = 0
    for t in range(T):
        tau2hat += (
            (y[0, t] - H @ x_pred1[:, t])**2 
            / (H @ V_pred1[:, :, t] @ H.T + R) / T).item()

    ll = -T * np.log(tau2hat) / 2 - T/2 - T/2 * np.log(2*np.pi)

    for t in range(T):
        ll -= np.log(H @ V_pred1[:, :, t] @ H.T + R) / 2

    mll = -ll.item()

    if not return_grad:
        grad = 0
        return mll, grad, tau2hat

    grad_F = np.zeros((2 * K, 2 * K, 3 * K))
    grad_Q = np.zeros((2 * K, 2 * K, 3 * K))

    for k in range(K):
        grad_F[2*k:2*k+2, 2*k:2*k+2, k] = (
            1 / 2 / (np.cosh(param[k])**2) * np.block([
                [np.cos(theta[k]), -np.sin(theta[k])], 
                [np.sin(theta[k]), np.cos(theta[k])]]))
        grad_F[2*k:2*k+2, 2*k:2*k+2, k + K] = (
            np.pi / (np.cosh(param[k + K])**2) * a[k] 
            * np.block([
                [-np.sin(theta[k]), -np.cos(theta[k])], 
                [np.cos(theta[k]), -np.sin(theta[k])]]))
        grad_Q[2*k:2*k+2, 2*k:2*k+2, 2*K + k] = sigma2[k] * np.eye(2)

    grad_x_pred1 = np.zeros((2 * K, 3 * K))
    grad_V_pred1 = np.zeros((2 * K, 2 * K, 3 * K))
    for k in range(K):
        grad_V_pred1[2*k:2*k+2, 2*k:2*k+2, k] = (
            a[k] / (np.cosh(param[k])**2) * sigma2[k] 
            / (1 - a[k]**2)**2 * np.eye(2))
        grad_V_pred1[2*k:2*k+2, 2*k:2*k+2, 2*K+k] = (
            sigma2[k] / (1 - a[k]**2) * np.eye(2))

    grad_x_filt = np.zeros((2 * K, 3 * K))
    grad_V_filt = np.zeros((2 * K, 2 * K, 3 * K))
    grad_Rhat = np.zeros((3 * K, 1))
    grad = np.zeros((3 * K, 1))
    for t in range(T):
        err = y[0, t] - H @ x_pred1[:, t]
        err_var = H @ V_pred1[:, :, t] @ H.T + R
        grad_err = (-H @ grad_x_pred1).T
        grad_err_var = np.zeros((3 * K, 1))

        grad_err_var[:, 0] = np.array(list(
            (H @ grad_V_pred1[:, :, i] @ H.T).item()
            for i in range(3*K)))

        for i in range(3*K):
            grad[i, 0] -= (
                H @ grad_V_pred1[:, :, i] @ H.T / err_var / 2).item()

            grad_Rhat[i, 0] -= (
                2 * (y[0, t] - H @ x_pred1[:, t]) 
                * H @ grad_x_pred1[:, i] / err_var / T).item()

            grad_Rhat[i, 0] -= (
                (y[0, t] - H @ x_pred1[:, t])**2 
                / (err_var**2) @ H @ grad_V_pred1[:, :, i] @ H.T / T).item()

        if t == (T - 1):
            break

        Kg = V_pred1[:, :, t] @ H.T / err_var
        grad_Kg = np.zeros((2 * K, 3 * K + 1))
        for i in range(3 * K):
            grad_Kg[:, i:i+1] = (
                grad_V_pred1[:, :, i] @ H.T / err_var 
                - V_pred1[:, :, t] @ H.T * grad_err_var[i, 0] / (err_var**2))
        
        for i in range(3*K):
            grad_x_filt[:, i] = (
                grad_x_pred1[:, i] 
                + (Kg * grad_err[i, 0]).squeeze() 
                + grad_Kg[:, i] * err)

            grad_V_filt[:, :, i] = (
                grad_V_pred1[:, :, i] 
                - grad_Kg[:, i:i+1] @ H @ V_pred1[:, :, t]
                - Kg @ H @ grad_V_pred1[:, :, i])

        for i in range(3*K):
            grad_x_pred1[:, i:i+1] = (
                F @ grad_x_filt[:, i:i+1] 
                + grad_F[:, :, i] @ x_filt[:, t:t+1])

            grad_V_pred1[:, :, i] = (
                F @ grad_V_filt[:, :, i] @ F.T 
                + grad_F[:, :, i] @ V_filt[:, :, t] @ F.T 
                + F @ V_filt[:, :, t] @ grad_F[:, :, i].T 
                + grad_Q[:, :, i])

    grad = grad - T * grad_Rhat / tau2hat / 2
    grad = -grad

    return mll, grad, tau2hat




