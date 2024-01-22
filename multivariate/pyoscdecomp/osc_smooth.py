import numpy as np
import numpy.linalg as LA

# Kalman smoother for oscillator model (both univariate and multivariate)
def osc_smooth(Y, fs: float, param):
    """
    Args:
        Y: ndarray, shape = (J, T)
        fs: float
        param: ndarray, shape = (1, (2*J+1)*K + 1))

    Returns:
        x_smooth: ndarray, shape = (2*K, T)
        V_smooth: ndarray, shape = (2*K, 2*K, T)
    """    
    J = Y.shape[0]
    T = Y.shape[1]
    K = (param.shape[1] - 1) // (3 + 2 * (J - 1))

    a = param[0, 0:K]
    theta = param[0, K:2*K] / fs * 2 * np.pi
    sigma2 = param[0, 2*K:3*K]
    c = param[0, 3*K:-1]
    tau2 = param[0, -1]
    
    F = np.zeros((2*K, 2*K))
    Q = np.zeros((2*K, 2*K))

    for k in range(K):
        F[2*k:2*k+2, 2*k:2*k+2] = (a[k] 
                                   * np.block([
                                    [np.cos(theta[k]), -np.sin(theta[k])], 
                                    [np.sin(theta[k]), np.cos(theta[k])]]))

        Q[2*k:2*k+2, 2*k:2*k+2] = sigma2[k] * np.eye(2)

    H = np.zeros((J, 2 * K))
    H[0, np.arange(0, 2 * K, 2)] = 1

    kk = 0
    for k in range(K):
        for j in range(1, J):
            H[j, 2*k:2*k+2] = c[kk:kk+2]
            kk += 2

    R = tau2 * np.eye(J)
    
    x_pred1 = np.zeros((2*K, T))
    x_filt = np.zeros((2*K, T))
    V_pred1 = np.zeros((2*K, 2*K, T))
    V_filt = np.zeros((2*K, 2*K, T))
    x_pred1[:, 0:1] = np.zeros((2*K, 1))

    for k in range(K):
        V_pred1[2*k:2*k+2, 2*k:2*k+2, 0] = (sigma2[k] / (1 - a[k]**2)
                                            * np.eye(2))

    for t in range(T):
        x_filt[:, t] = (x_pred1[:, t] 
                        + (V_pred1[:, :, t] @ H.T 
                           @ LA.solve(
                             H @ V_pred1[:, :, t] @ H.T + R, 
                             Y[:, t] - H @ x_pred1[:, t])))
        
        V_filt[:, :, t] = (V_pred1[:, :, t] 
                           - V_pred1[:, :, t] @ H.T
                           @ LA.solve(H@V_pred1[:, :, t]@H.T + R, H)
                           @ V_pred1[:, :, t])

        if (t == T-1):
            break

        x_pred1[:, t + 1] = F @ x_filt[:, t]
        V_pred1[:, :, t + 1] = F @ V_filt[:, :, t] @ F.T + Q

    x_smooth = np.zeros((2*K, T))
    V_smooth = np.zeros((2*K, 2*K, T))

    x_smooth[:, T - 1] = x_filt[:, T - 1]
    V_smooth[:, :, T - 1] = V_filt[:, :, T - 1]

    for t in range(T-2, -1, -1):
        x_smooth[:, t] = (x_filt[:, t]
                          + V_filt[:, :, t] @ F.T @ LA.solve(
                            V_pred1[:, :, t + 1], 
                            x_smooth[:, t + 1] - x_pred1[:, t + 1]))
        V_smooth[:, :, t] = (V_filt[:, :, t]
                             + V_filt[:, :, t] @ F.T 
                             @ LA.solve(
                                V_pred1[:, :, t + 1], 
                                V_smooth[:, :, t + 1] - V_pred1[:, :, t + 1]) 
                             @ LA.solve(V_pred1[:, :, t + 1], F) 
                             @ V_filt[:, :, t])

    return x_smooth, V_smooth