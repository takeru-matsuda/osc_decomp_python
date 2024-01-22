import numpy as np
import numpy.linalg as LA
import warnings
from scipy.optimize import minimize, BFGS, Bounds
from typing import Optional
from .pyoscdecompuni.AR_fit import AR_fit
from .pyoscdecompuni.whittle_uni_fit import whittle_uni_fit
from .pyoscdecompuni.osc_uni_prof_ll import osc_uni_prof_ll
from .osc_smooth import osc_smooth


def osc_decomp_uni(
        y,
        fs: float, 
        MAX_OSC: Optional[int] = None,
        MAX_AR: Optional[int] = None,
        algorithm: Optional[str] = None,
        grad: Optional[bool] = None):
    """
    Args:
        y: ndarray, shape = (1, T)
            univariate time series (1 times T)  
        fs: float
            sampling frequency (scalar)
        MAX_OSC: int Defaults to None.
            maximum number of oscillation components (scalar)
        MAX_AR: int Defaults to None.
            maximum AR degree (scalar), 
            set to be 2 * MAX_OSC when None (default).
        algorithm: str Defaults to None.
            optimization algorithm (string)
            set to be 'quasi-newton' when None (default).
        grad: bool Defaults to None.
            gradient computation method in optimization (boolean), 
            set to be False when None (default).
            True -> Kalman filter, False -> numerical gradient
    Returns:
        osc_param: ndarray, shape = (MAX_OSC, 3 * MAX_OSC + 1)
            estimated parameters of the oscillator model 
            (MAX_OSC times 3*MAX_OSC+1)
            osc_param[K - 1, 0 : K] is the estimated a_1,...,a_K of 
            the oscillator model with K oscillation components
            osc_param[K - 1, K : 2*K] is the estimated f_1,...f_K of 
            the oscillator model with K oscillation components
            osc_param[K - 1, 2*K+1 : 3*K] is 
            the estimated sigma_1^2,...,sigma_K^2 
            of the oscillator model with K oscillation components
            osc_param[K - 1, 3*K + 1] is the estimated tau^2 of 
            the oscillator model with K oscillation components
        osc_AIC: ndarray, shape = (1, MAX_OSC)        
            AIC_osc[K - 1] is the AIC of the oscillator model 
            with K oscillation components
        osc_mean: ndarray, shape = (2 * MAX_OSC, T, MAX_OSC)
            smoothed coordinate of each oscillator 
            (2*MAX_OSC times T times MAX_OSC)
            osc_mean[2*k : 2*k+2, :, K - 1] (k starts from 0) is 
            the smoothed coordinate of the (k+1)-th oscillator 
            in the decomposition into K components
        osc_cov: ndarray, shape = (2 * MAX_OSC, 2 * MAX_OSC, T, MAX_OSC)
            smoothed covariance of each oscillator 
            (2*MAX_OSC times 2*MAX_OSC times T times MAX_OSC)
            osc_cov[2*k : 2*k+2, 2*k : 2*k+2, :, K - 1] (k starts from 0) is 
            the smoothed covariance of 
            the (k+1)-th oscillator in the decomposition into K components
        osc_phase: ndarray, shape = (MAX_OSC, T, MAX_OSC)
            estimated phase of each oscillation component 
            (MAX_OSC times T times MAX_OSC)
            osc_phase[k, :, K - 1] (k starts from 0) is the estimated phase of 
            the (k+1)-th oscillator 
            in the decomposition into K components
    """                       

    T = y.shape[1]

    if (MAX_OSC is None):
        MAX_OSC = 5

    if (MAX_AR is None):
        MAX_AR = max(20, 2 * MAX_OSC)
    else:   
        MAX_AR = max(MAX_AR, 2 * MAX_OSC)

    if (grad is None):
        grad = False

    [ARwithnoise_param,
     ARwithnoise_AIC,
     AR_param,
     AR_AIC] = AR_fit(y, MAX_AR)
    osc_param = np.zeros((MAX_OSC, 3 * MAX_OSC + 1))
    osc_AIC = np.zeros((1, MAX_OSC))
    osc_mean = np.zeros((2 * MAX_OSC, T, MAX_OSC))
    osc_cov = np.zeros((2 * MAX_OSC, 2 * MAX_OSC, T, MAX_OSC))
    osc_phase = np.zeros((MAX_OSC, T, MAX_OSC))

    for K in range(1, MAX_OSC + 1):
        ARdeg = np.argmin(ARwithnoise_AIC) + 1
        tmp = np.roots(
            np.concatenate([
                np.ones(1), 
                ARwithnoise_param[ARdeg - 1, 0:ARdeg]]))       

        if (np.count_nonzero(tmp.imag >= 0.0) >= K):
            z0 = tmp
            E0 = ARwithnoise_param[ARdeg - 1, ARdeg]
            R0 = ARwithnoise_param[ARdeg - 1, ARdeg + 1]
            optARdeg = ARdeg
        else:
            minAIC = np.inf
            minAIC2 = np.inf
            for ARdeg in range(K, 2 * K + 1):
                tmp = np.roots(
                    np.concatenate([
                        np.ones(1), 
                        ARwithnoise_param[ARdeg - 1, 0:ARdeg]]))

                if (((ARdeg - np.count_nonzero(tmp.imag)/2) == K) and
                        (ARwithnoise_AIC[0, ARdeg - 1] < minAIC)):
                    z0 = tmp
                    E0 = ARwithnoise_param[ARdeg - 1, ARdeg]
                    R0 = ARwithnoise_param[ARdeg - 1, ARdeg + 1]
                    minAIC = ARwithnoise_AIC[0, ARdeg - 1]
                    optARdeg = ARdeg

                if (((ARdeg - np.count_nonzero(tmp.imag)/2) == K) and
                        (ARwithnoise_AIC[0, ARdeg - 1] < minAIC2)):
                    z1 = tmp
                    E1 = ARwithnoise_param[ARdeg - 1, ARdeg]
                    R1 = ARwithnoise_param[ARdeg - 1, ARdeg + 1]
                    minAIC2 = ARwithnoise_AIC[0, ARdeg - 1]
                    optARdeg2 = ARdeg

            if (minAIC == np.inf):
                if (minAIC2 == np.inf):
                    warnings.warn('no AR model with {} oscillators'.format(K))

                z0 = z1
                E0 = E1
                R0 = R1
                optARdeg = optARdeg2
                I = np.argsort(-np.abs(z0))
                z0 = z0[I]

        VV = np.zeros((optARdeg, optARdeg)).astype(complex)
        VV = np.array([
            [z0[j]**(-i) for j in range(optARdeg)] 
            for i in range(optARdeg)])

        QQ = (
            LA.inv(VV)
            @ np.block([
                [E0, np.zeros((1, optARdeg - 1))],
                [np.zeros((optARdeg - 1, optARdeg))]])
            @ np.conjugate(LA.inv(VV).T))

        I = np.argsort(-(np.diag(QQ.real) / (1 - np.abs(z0)**2)))

        z0 = z0[I]

        init_a = np.zeros((1, K))
        init_theta = np.zeros((1, K))
        kk = 0
        for k in range(K):
            init_a[0, k] = np.abs(z0[kk])
            init_theta[0, k] = np.abs(np.angle(z0[kk]))
            if (z0[kk].imag == 0):
                kk += 1
            else:
                kk += 2

        if (np.mod(T, 2) == 0):
            freq = np.hstack(
                [2 * np.pi / T * np.arange(T/2), np.pi])
        else:
            freq = np.hstack(
                [2 * np.pi / T * np.arange((T-1)/2 + 1)])

        P = np.zeros((freq.size, K + 1))
        for k in range(K):
            a = init_a[0, k]
            theta = init_theta[0, k]
            A = (
                (1 - 2 * a**2 * np.cos(theta)**2 + a**4 * np.cos(2*theta))
                / a / (a**2 - 1) / np.cos(theta))

            b = ((
                A - 2 * a * np.cos(theta) 
                + np.sign(np.cos(theta)) * np.sqrt(
                    (A - 2 * a * np.cos(theta))**2 - 4)) / 2)

            for j in range(freq.size):
                P[j, k] = (
                    -a * np.cos(theta)
                    / b * np.abs(1 + b * np.exp(-1j*freq[j]))**2
                    / np.abs(
                        1 - 2 * a * np.cos(theta) * np.exp(-1j*freq[j]) 
                        + a**2 * np.exp(-2*1j*freq[j]))**2
                    / 2 / np.pi)

        P[:, K] = 1 / 2 / np.pi
        p = np.empty((freq.size, 1))

        for j in range(freq.size):
            p[j] = (np.abs(
                y @ np.exp(-1j * freq[j] * np.arange(T).reshape(T, 1)))**2
                / 2 / np.pi / T)

        weight = whittle_uni_fit(P, p)
        init_sigma2 = weight[0:K, 0:1].T
        init_tau2 = weight[K, 0]
        init_param = np.hstack([
            np.arctanh(2 * init_a[0, :] - 1), 
            np.zeros(K), 
            np.log(init_sigma2 / init_tau2).reshape(K)])

        if (algorithm is not None):
            if (algorithm.lower() == 'quasi-newton'):
                if (grad):
                    return_grad = True
                    res = minimize(
                        (lambda param, y, init_theta, return_grad:
                            osc_uni_prof_ll(
                                y, param, init_theta, False)[0]),
                        init_param,
                        jac=(
                            lambda param, y, init_theta, return_grad: 
                            osc_uni_prof_ll(
                                y, param,
                                init_theta, return_grad)[1].squeeze()),
                        method="BFGS",
                        args=(y, init_theta, return_grad,),
                        tol=1e-6
                    )
                    param = res.x.reshape((1, 3 * K))
                else:
                    return_grad = False
                    res = minimize(
                        (lambda param, y, init_theta, return_grad:
                            osc_uni_prof_ll(
                                y, param, init_theta, return_grad)[0]),
                        init_param,
                        method='BFGS',
                        args=(y, init_theta, return_grad,),
                        tol=1e-6
                    )
                    param = res.x.reshape((1, 3 * K))
            elif (algorithm.lower() == 'trust-region'):
                return_grad = True
                res = minimize(
                    (lambda param, y, init_theta, return_grad:
                        osc_uni_prof_ll(y, param, init_theta, False)[0]),
                    init_param,
                    method="trust-constr",
                    args=(y, init_theta, return_grad,),
                    jac=(
                        lambda param, y, init_theta, return_grad:
                            osc_uni_prof_ll(
                                y, param,
                                init_theta, return_grad)[1].squeeze()),
                    hess=BFGS(),
                    options={'gtol': 1e-6,
                             'xtol': 1e-6,
                             'finite_diff_rel_step': 1e-6,
                             'maxiter': 1000}
                )
                param = res.x.reshape((1, 3 * K))
            else:
                raise ValueError('osc_decomp_uni: unknown algorithm')
        else:
            return_grad = False
            res = minimize(
                (lambda param, y, init_theta, return_grad:
                    osc_uni_prof_ll(y, param, init_theta, return_grad)[0]),
                init_param,
                method='BFGS',
                args=(y, init_theta, return_grad,),
                options={},
                tol=1e-6
            )
            param = res.x.reshape((1, 3*K))

        [mll, _, osc_tau2] = osc_uni_prof_ll(y, param.squeeze(), 
                                               init_theta, False)
        osc_AIC[0, K - 1] = 2 * mll + 2 * (3 * K + 1)
        param[0, K:2*K] = init_theta + np.tanh(param[0, K:2*K]) * np.pi
        I = np.argsort(np.abs(param[0, K:2*K]))
        osc_a = (np.tanh(param[0:1, I]) + 1) / 2
        osc_f = np.abs(param[0:1, K + I]) * fs / 2 / np.pi
        osc_sigma2 = np.exp(param[0:1, 2 * K + I]) * osc_tau2
        osc_param[K - 1, 0:3*K+1] = np.hstack([
            osc_a, osc_f, osc_sigma2, np.reshape(osc_tau2, (1, 1))])

        [osc_mean[0:2*K, :, K - 1], 
         osc_cov[0:2*K, 0:2*K, :, K-1]] = osc_smooth(
            y, fs, osc_param[K-1:K, 0:3*K+1])

        for k in range(K):
            osc_phase[k, :, K - 1] = np.arctan2(
                osc_mean[2 * k + 1, :, K - 1],
                osc_mean[2 * k, :, K - 1])

    return osc_param, osc_AIC, osc_mean, osc_cov, osc_phase
