import numpy as np
import numpy.linalg as LA


def AR_ll(y, p):
    """
    Args:
        y: ndarray, shape = (1, T)
        p: ndarray, shape = (1, ARdeg)

    Returns:
        mll: float
        Ehat: float
    """    
    T = y.shape[1]
    ARdeg = p.shape[1]
    c = (np.exp(p[0, :]) - 1) / (np.exp(p[0, :]) + 1)
    a = np.zeros((ARdeg, ARdeg))

    np.fill_diagonal(a, c)

    if (ARdeg >= 2):
        a[0, 1] = a[0, 0] - c[1] * a[0, 0]
        for m in range(1, ARdeg):
            for i in range(m):
                a[i, m] = a[i, m - 1] - c[m] * a[m - i - 1, m - 1]
                
    K = np.zeros((ARdeg + 1, ARdeg + 1))
    A = np.concatenate([np.ones(1), -a[:, ARdeg - 1].T])
    K[0, 0:ARdeg+1] += A[0:ARdeg + 1]
    for i in range(1, ARdeg + 1):
        K[i, np.arange(i, 0, -1)] += A[0:i]
        K[i, 0:ARdeg-i+1] += A[i:ARdeg+1]

    C = LA.solve(K, 
                 np.concatenate([np.ones(1), 
                                 np.zeros(ARdeg)])).reshape([ARdeg + 1, 1])
    F = np.zeros((ARdeg, ARdeg))
    F[:, 0] = a[:, ARdeg - 1]
    F[:ARdeg-1, 1:ARdeg] = np.eye(ARdeg - 1)
    G = np.concatenate([np.ones((1, 1)), np.zeros((ARdeg - 1, 1))])
    Q = np.dot(G, G.T)
    H = np.concatenate([np.ones((1, 1)), np.zeros((1, ARdeg - 1))], axis=1)
    R = 0
    x_pred1 = np.zeros((ARdeg, T))
    x_filt = np.zeros((ARdeg, T))
    V_pred1 = np.zeros((ARdeg, ARdeg, T))
    V_filt = np.zeros((ARdeg, ARdeg, T))
    x_pred1[:, 0:1] = np.zeros((ARdeg, 1))
    V_pred1[0, 0, 0] = C[0]
    for i in range(1, ARdeg):
        V_pred1[i, 0, 0] = np.dot(C[1:ARdeg-i+1].T,
                                  a[i:ARdeg, ARdeg - 1])
        V_pred1[0, i, 0] = V_pred1[i, 0, 0]
 
    for i in range(1, ARdeg):
        for j in range(i, ARdeg):
            for p in range(i, ARdeg):
                for q in range(j, ARdeg):
                    V_pred1[i, j, 0] += (a[p, ARdeg - 1]
                                         * a[q, ARdeg - 1]
                                         * C[np.abs(q - j - p + i)])
            V_pred1[j, i, 0] = V_pred1[i, j, 0]

    for t in range(T - 1):        
        x_filt[:, t] = x_pred1[:, t] + (V_pred1[:, :, t] 
                                        @ H.T 
                                        @ LA.solve(
                                            H @ V_pred1[:, :, t] @ H.T + R,
                                            y[0, t] - H @ x_pred1[:, t]))

        V_filt[:, :, t] = ((np.eye(ARdeg) 
                            - (V_pred1[:, :, t] @ (H.T @ H)) 
                            / (H @ V_pred1[:, :, t] @ H.T + R))
                           @ V_pred1[:, :, t])

        x_pred1[:, t + 1] = F @ x_filt[:, t]
        V_pred1[:, :, t + 1] = F @ V_filt[:, :, t] @ F.T + Q    

    Ehat = 0
    for t in range(T):
        Ehat = (Ehat * t
                + (y[0, t] - H @ x_pred1[:, t]).item()**2
                / (H @ V_pred1[:, :, t] @ H.T).item()) / (t + 1)

    ll = - T * np.log(Ehat) / 2 - T/2 - T/2 * np.log(2*np.pi)
    
    for t in range(T):
        ll -= (np.log(H @ V_pred1[:, :, t] @ H.T) / 2).item()

    mll = -ll

    return mll, Ehat
