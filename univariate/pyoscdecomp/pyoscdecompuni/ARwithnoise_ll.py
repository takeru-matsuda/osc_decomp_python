import numpy as np
import numpy.linalg as LA
from scipy.linalg import toeplitz


def ARwithnoise_ll(y, param):
    """
    Args:
        y: ndarray, shape = (1, T)
        param: ndarray, shape = (1, ARdeg + 1)

    Returns:
        mll: float
        Rhat: float
    """    
    T = y.shape[1]
    ARdeg = param.shape[1] - 1

    A = np.concatenate([np.ones((1, 1)), param[0:1, 0:ARdeg]], axis=1)
    E = np.exp(param[0, ARdeg])
    R = 1
    x_pred1 = np.zeros((ARdeg, T))
    x_filt = np.zeros((ARdeg, T))
    V_pred1 = np.zeros((ARdeg, ARdeg, T))
    V_filt = np.zeros((ARdeg, ARdeg, T))

    F = np.concatenate([-A[0:1, 1:ARdeg+1], 
                        np.concatenate(
                            [np.eye(ARdeg - 1),
                             np.zeros((ARdeg - 1, 1))], axis=1)], axis=0)
    Q = np.concatenate([
        np.concatenate(
            [E * np.ones((1, 1)), np.zeros((1, ARdeg - 1))], axis=1),
        np.zeros((ARdeg - 1, ARdeg))], axis=0)
    H = np.concatenate([np.ones((1, 1)), np.zeros([1, ARdeg - 1])], axis=1)

    x_pred1[:, 0:1] = np.zeros((ARdeg, 1))
    K = np.zeros((ARdeg + 1, ARdeg + 1))

    K[0, 0:ARdeg+1] += A[0, 0:ARdeg+1]
    for i in range(1, ARdeg + 1):
        K[i, np.arange(i, 0, -1)] += A[0, 0:i]
        K[i, 0:ARdeg-i+1] += A[0, i:ARdeg+1]

    c = LA.solve(
        K, np.concatenate(
            [E * np.ones((1, 1)), np.zeros((ARdeg, 1))], axis=0))
    V_pred1[:, :, 0] = toeplitz(c[0:ARdeg])
    for t in range(T - 1):        
        x_filt[:, t] = x_pred1[:, t] + (
            V_pred1[:, :, t] @ H.T 
            @ LA.solve(
                H @ V_pred1[:, :, t] @ H.T + R, y[0, t] - H @ x_pred1[:, t]))

        V_filt[:, :, t] = (
            (np.eye(ARdeg) - (V_pred1[:, :, t] @ (H.T @ H)) 
             / (H @ V_pred1[:, :, t] @ H.T + R)) @ V_pred1[:, :, t])

        x_pred1[:, t + 1] = F @ x_filt[:, t]
        V_pred1[:, :, t + 1] = F @ V_filt[:, :, t] @ F.T + Q    
    Rhat = 0
    for t in range(T):
        Rhat = (
            (Rhat * t 
             + ((y[0, t] - H @ x_pred1[:, t])).item()**2 
             / ((H @ V_pred1[:, :, t] @ H.T + R)).item()) / (t + 1))

    ll = (- T * np.log(Rhat) / 2 - T/2 - T/2 * np.log(2*np.pi)).item()

    for t in range(T):
        ll -= (np.log(H @ V_pred1[:, :, t] @ H.T + R) / 2).item()

    mll = -ll

    return mll, Rhat


