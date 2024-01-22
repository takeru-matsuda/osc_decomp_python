import numpy as np
import numpy.linalg as LA
import copy

# minus log-likelihood of stationary VAR model with observation noise
def VARwithnoise_ll(Y, A, E, R):    
    """
    Args:
        Y: ndarray, shape = (J, T)
        A: ndarray, shape = (J, J * ARdeg)
        E: ndarray, shape = (J, J)
        R: ndarray, shape = (J, J)

    Returns:
        mll: float
    """    
    EPS = 1e-6
    J = Y.shape[0]
    T = Y.shape[1]
    ARdeg = A.shape[1] // J
    x_pred1 = np.zeros((J*ARdeg, T))
    x_filt = np.zeros((J*ARdeg, T))
    V_pred1 = np.zeros((J*ARdeg, J*ARdeg, T))
    V_filt = np.zeros((J*ARdeg, J*ARdeg, T))
    F = np.block(
        [[A[:, 0:J*ARdeg]], 
         [np.eye(J*(ARdeg-1)), np.zeros((J*(ARdeg-1), J))]])
    Q = np.block(
        [[E, np.zeros((J, J*(ARdeg-1)))], 
         [np.zeros((J*(ARdeg-1),J*ARdeg))]])
    H = np.block(
        [np.eye(J), np.zeros((J, J*(ARdeg-1)))])
    x_pred1[:, 0] = np.zeros(J*ARdeg)
    V_pred1[:, :, 0] = F @ Q @ F.T + Q    
    prev = Q
    cnt = 0

    while (
        (LA.norm(V_pred1[:, :, 0] - prev, 'fro') 
         / LA.norm(prev, 'fro')) > EPS):
        prev = copy.deepcopy(V_pred1[:, :, 0])
        V_pred1[:, :, 0] = F @ V_pred1[:, :, 0] @ F.T + Q
        cnt += 1

    for t in range(T-1):
        x_filt[:, t:t+1] = (
            x_pred1[:, t:t+1] 
            + V_pred1[:, :, t] @ H.T @ LA.solve(
                H @ V_pred1[:, :, t] @ H.T + R, 
                Y[:, t:t+1] - H @ x_pred1[:, t:t+1]))
        V_filt[:, :, t] = (
            V_pred1[:, :, t] 
            - V_pred1[:, :, t] @ H.T @ LA.solve(
                H @ V_pred1[:, :, t] @ H.T + R, H) 
            @ V_pred1[:, :, t])
        x_pred1[:, t+1:t+2] = F @ x_filt[:, t:t+1]
        V_pred1[:, :, t+1] = F @ V_filt[:, :, t] @ F.T + Q

    ll = -J * T/2 * np.log(2*np.pi)
    for t in range(T):
        ll -= (
            np.log(LA.det(H @ V_pred1[:, :, t] @ H.T + R))/2 
            + (Y[:, t:t+1] - H @ x_pred1[:, t:t+1]).T @ LA.solve(
                H @ V_pred1[:, :, t] @ H.T + R, 
                Y[:, t:t+1] - H @ x_pred1[:, t:t+1])/2).item()

    # for minimization
    mll = -ll
    return mll

