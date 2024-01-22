import numpy as np
import numpy.linalg as LA
import warnings
from math import ceil
from scipy.optimize import minimize, BFGS
from typing import Optional
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
sys.path.append(current_dir)
from pyoscdecomp.pyoscdecompmulti.VAR_fit import VAR_fit
from pyoscdecomp.pyoscdecompmulti.whittle_multi_fit import whittle_multi_fit
from pyoscdecomp.pyoscdecompmulti.osc_multi_prof_ll import osc_multi_prof_ll
from pyoscdecomp.pyoscdecompmulti.polyeig_VAR import polyeig_VAR
from pyoscdecomp.osc_smooth import osc_smooth


def osc_decomp_multi(
    Y,
    fs: float,
    MAX_OSC: Optional[int] = None,
    MAX_VAR: Optional[int] = None,
    algorithm: Optional[str] = None,
    grad: Optional[bool] = None):
    """
    Args:
        Y: ndarray, shape = (J, T)
            multivariate time series (J times T)
        fs: float
            sampling frequency (scalar)
        MAX_OSC: int Defaults to None.
            maximum number of oscillation components (scalar)        
        MAX_VAR: int Defaults to None.
            maximum VAR degree (scalar)
            set to be max(20, ceil(2*MAX_OSC/J)) when None (default).
        algorithm: str Defaults to None.
            optimization algorithm (string)
            set to be 'trust-region-ncg' when None (default).
        grad: bool Defaults to None.
            gradient computation method in optimization (boolean), 
            set to be False when None (default).
            True -> Kalman filter, False -> numerical gradient
    Returns:
        osc_param: ndarray, shape = (MAX_OSC, (2*J+1)*MAX_OSC + 1)
            estimated parameters of the oscillator model 
            (MAX_OSC times (2*J+1)*MAX_OSC+1)
            osc_param[K - 1, 0:K] is the estimated a_1,...,a_K of 
            the oscillator model with K oscillation components
            osc_param[K - 1, K:2*K] is the estimated f_1,...f_K of 
            the oscillator model with K oscillation components
            osc_param[K - 1, 2*K+1:3*K] is 
            the estimated sigma_1^2,...,sigma_K^2 
            osc_param[K - 1, 3*K:(2*J+1)*K] is the estimated 
            c_{21,1},c_{21,2},...,c_{J1,1},c_{J1,2},...,c_{JK,1},c_{JK,2} 
            of the oscillator model with K oscillation components            
            osc_param[K - 1, (2*J+1)*K] is the estimated tau^2 of 
            the oscillator model with K oscillation components
        osc_AIC: ndarray, shape = (1, MAX_OSC)        
            osc_AIC[0, K - 1] is the AIC of the oscillator model 
            with K oscillation components
        osc_mean: ndarray, shape = (2 * MAX_OSC, T, MAX_OSC)
            smoothed coordinate of each oscillator 
            (2*MAX_OSC times T times MAX_OSC)
            osc_mean[2*k:2*k+2, :, K - 1] (k starts from 0) is 
            the smoothed coordinate of the (k+1)-th oscillator 
            in the decomposition into K components
        osc_cov: ndarray, shape = (2 * MAX_OSC, 2 * MAX_OSC, T, MAX_OSC)
            smoothed covariance of each oscillator 
            (2*MAX_OSC times 2*MAX_OSC times T times MAX_OSC)
            osc_cov[2*k:2*k+2, 2*k:2*k+2, :, K - 1] (k starts from 0) is 
            the smoothed covariance of 
            the (k+1)-th oscillator in the decomposition into K components
        osc_phase: ndarray, shape = (MAX_OSC, T, MAX_OSC)
            estimated phase of each oscillation component 
            (MAX_OSC times T times MAX_OSC)
            osc_phase[k, :, K - 1] (k starts from 0) is the estimated phase of 
            the (k+1)-th oscillator 
            in the decomposition into K components
    """    
    J = Y.shape[0]
    T = Y.shape[1]
    if (MAX_OSC is None):
        MAX_OSC = 5

    if (MAX_VAR is None):
        MAX_VAR = max(20, ceil(2*MAX_OSC/J))
    else:   
        MAX_VAR = max(MAX_VAR, ceil(2*MAX_OSC/J))

    if (algorithm is None):
        # Default to Newton conjugate gradient trust-region algorithm.
        algorithm = 'trust-region-ncg'

    if (grad is None):
        grad = False

    [VARwithnoise_A,
     VARwithnoise_E,
     VARwithnoise_r,
     VARwithnoise_AIC] = VAR_fit(Y, MAX_VAR)

    osc_param = np.zeros((MAX_OSC, (2*J+1)*MAX_OSC+1))
    osc_AIC = np.zeros((1, MAX_OSC))
    osc_mean = np.zeros((2*MAX_OSC, T, MAX_OSC))
    osc_cov = np.zeros((2*MAX_OSC, 2*MAX_OSC, T, MAX_OSC))
    osc_phase = np.zeros((MAX_OSC, T, MAX_OSC))
    for K in range(1, MAX_OSC+1):
        print('K = ', K)
        ARdeg = np.nanargmin(VARwithnoise_AIC) + 1
        [Vtmp, tmp] = polyeig_VAR(
            VARwithnoise_A[:, 0:J*ARdeg, ARdeg-1])
        if (np.count_nonzero(tmp.imag >= 0.0) >= K):
            V0 = Vtmp
            z0 = tmp
            E0 = VARwithnoise_E[:, :, ARdeg-1]
            R0 = VARwithnoise_r[0, ARdeg-1]
            minAIC = VARwithnoise_AIC[0, ARdeg-1]
            optARdeg = ARdeg
        else:
            minAIC = np.inf
            minAIC2 = np.inf
            minK = np.inf
            for ARdeg in range(ceil(K/J), MAX_VAR+1):
                [Vtmp, tmp] = polyeig_VAR(
                    VARwithnoise_A[:, 0:J*ARdeg, ARdeg-1])

                if (
                    (
                        (J*ARdeg - np.count_nonzero(tmp.imag)/2) == K) 
                        and (VARwithnoise_AIC[0, ARdeg-1] < minAIC)):
                    V0 = Vtmp
                    z0 = tmp
                    E0 = VARwithnoise_E[:, :, ARdeg-1]
                    R0 = VARwithnoise_r[0, ARdeg-1]
                    minAIC = VARwithnoise_AIC[0, ARdeg-1]
                    optARdeg = ARdeg

                if (
                    (J*ARdeg - np.count_nonzero(tmp.imag)/2 > K) 
                    and 
                    ((J*ARdeg-np.count_nonzero(tmp.imag)/2 < minK)
                     or (J*ARdeg-np.count_nonzero(tmp.imag)/2 == minK 
                         and VARwithnoise_AIC[0, ARdeg-1] < minAIC2))
                    ):
                    V1 = Vtmp
                    z1 = tmp
                    E1 = VARwithnoise_E[:, :, ARdeg-1]
                    R1 = VARwithnoise_r[0, ARdeg-1]
                    minAIC2 = VARwithnoise_AIC[0, ARdeg-1]
                    optARdeg2 = ARdeg
                    minK = J*ARdeg - np.count_nonzero(tmp.imag)/2

            if minAIC == np.inf:
                if minAIC2 == np.inf:
                    warnings.warn(
                        'no VAR model with {} oscillators'.format(K))
                V0 = V1
                z0 = z1
                E0 = E1
                R0 = R1
                optARdeg = optARdeg2

        VV = np.zeros((J*optARdeg, J*optARdeg)).astype(complex)
        for j in range(J*optARdeg):
            for i in range(optARdeg):
                VV[i*J:(i+1)*J, j] = z0[j]**(-i) * V0[:, j]

        QQ = LA.inv(VV) @ np.block(
            [[E0, np.zeros((J, J*(optARdeg-1)))], 
             [np.zeros((J*(optARdeg-1), J*optARdeg))]]) @ np.conjugate(LA.inv(VV).T)
        I = np.argsort(-(np.diag(QQ.real) / (1 - np.abs(z0)**2)))
        V0 = V0[:, I]
        z0 = z0[I]

        init_a = np.zeros((1, K))
        init_theta = np.zeros((1, K))
        init_c = np.zeros((1, 2*(J-1)*K))
        kk = 0
        for k in range(K):
            init_a[0, k] = np.abs(z0[kk])
            init_theta[0, k] = np.abs(np.angle(z0[kk]))
            for j in range(J-1):
                init_c[0, 2*(J-1)*k+2*j] = (V0[j+1, kk]/V0[0, kk]).real
                if ((z0[kk]).imag < 0):
                    init_c[0, 2*(J-1)*k+2*j+1] = (V0[j+1, kk]/V0[0, kk]).imag
                else:
                    init_c[0, 2*(J-1)*k+2*j+1] = -(V0[j+1, kk]/V0[0, kk]).imag

            if (z0[kk]).imag == 0:
                kk += 1
            else:
                kk += 2

        if (np.mod(T, 2) == 0):
            freq = 2 * np.pi / T * np.arange(0, T/2 + 1)
        else:
            freq = 2 * np.pi / T * np.arange(0, (T-1)/2 + 1)

        P = np.zeros((J, J, len(freq), K+1)).astype(complex)
        for k in range(K):
            a = init_a[0, k]
            theta = init_theta[0, k]
            H = np.block(
                [[1, 0],
                 [init_c[0, 2*(J-1)*k:2*(J-1)*(k+1)].reshape(J-1, 2)]])
            for i in range(len(freq)):
                A = np.abs(1 - a * np.exp(1j*(theta - freq[i])))**(-2)
                B = np.abs(1 - a * np.exp(1j*(theta + freq[i])))**(-2)
                P[:, :, i, k] = H @ np.block(
                    [[A+B, 1j*(A-B)], 
                     [-1j*(A-B), A+B]]) @ np.conjugate(H.T) / 4 / np.pi

        for i in range(len(freq)):
            P[:, :, i, K] = np.eye(J) / 2 / np.pi

        p = np.zeros((J, len(freq))).astype(complex)
        for i in range(len(freq)):
            p[:, i] = (
                Y @ np.exp(-1j*freq[i] * np.arange(1, T+1).T) 
                / np.sqrt(T))
        weight = whittle_multi_fit(P, p)
        init_sigma2 = weight[0:K, 0:1].T
        init_tau2 = weight[K, 0]
        if algorithm.lower() == 'trust-region-ncg':
            # for 'trust-region-ncg' case, initial value for (transformed) osc_f is eps
            init_param = np.block(
                [np.arctanh(2*init_a-1), 
                 np.finfo(float).eps * np.ones((1, K)), 
                 np.log(init_sigma2/init_tau2), 
                 init_c]).squeeze()
        else:
            init_param = np.block(
                [np.arctanh(2*init_a-1), 
                 np.zeros((1, K)), 
                 np.log(init_sigma2/init_tau2), 
                 init_c]).squeeze()

        # Objective function
        func = (
            lambda param, Y, init_theta:
            osc_multi_prof_ll(Y, param, init_theta, False)[0])
        # Gradient
        if grad or (algorithm.lower() != 'quasi-newton'):
            jac = (
                lambda param, Y, init_theta:
                osc_multi_prof_ll(Y, param, init_theta, True)[1].squeeze())
        else:
            jac = None

        if (algorithm.lower() == 'quasi-newton'):
            bfgs_mxitr_exceed = 1
            exitflag = False
            while exitflag == False:
                res = minimize(
                    func,
                    init_param,
                    jac=jac,
                    method='BFGS',
                    args=(Y, init_theta),
                    options={
                        'maxiter': 400, 
                        'gtol': 1e-6, 
                        },
                )
                init_param = res.x
                exitflag = (res.status != bfgs_mxitr_exceed)
                if not exitflag:
                    print('exitflag is False: restart')
                    init_param = res.x
            param = res.x.reshape((1, (2*J+1)*K))
            fval = res.fun
        elif (algorithm.lower() == 'trust-region'):
            res = minimize(
                func,
                init_param,
                jac=jac,
                hess=BFGS(),
                method='trust-constr',
                args=(Y, init_theta),
                options={
                    'gtol': 1e-6,
                    'xtol': 1e-6,
                    'finite_diff_rel_step': 1e-6,
                    'maxiter': 1000}
            )
            param = res.x.reshape((1, (2*J+1)*K))
            fval = res.fun
        elif (algorithm.lower() == 'trust-region-ncg'):
            res = minimize(
                func,
                init_param,
                jac=jac,
                hess=BFGS(),
                method='trust-ncg',
                args=(Y, init_theta),
                options={
                    'gtol': 1e-6,
                    'maxiter': 1000},
            )
            param = res.x.reshape((1, (2*J+1)*K))
            fval = res.fun
        else:
            raise ValueError('osc_decomp_multi: unknown algorithm')

        [mll, _, osc_tau2] = osc_multi_prof_ll(
            Y, 
            param.squeeze(), 
            init_theta, 
            False)

        osc_AIC[0, K-1] = 2*mll + 2*((2*J+1)*K + 1)
        param[0, K:2*K] = (
            init_theta.squeeze() + np.tanh(param[0, K:2*K]) * np.pi)
        I = np.argsort(np.abs(param[0, K:2*K]))
        osc_a = (np.tanh(param[0:1, I]) + 1) / 2
        osc_f = np.abs(param[0:1, K+I]) * fs / 2 / np.pi
        osc_sigma2 = np.exp(param[0:1, 2*K+I]) * osc_tau2
        itmp = np.zeros((1, 2*(J-1)*K)).astype(int)
        for k in range(K):
            itmp[0, k*2*(J-1):(k+1)*2*(J-1)] = I[k]
        ind_osc_c = 3*K + itmp*2*(J-1) + np.tile(
            np.arange(0, 2*(J-1), dtype=int), K)
        osc_c = param[0:1, ind_osc_c.squeeze()]
        osc_param[K-1:K, 0:(2*J+1)*K+1] = np.block(
            [osc_a, osc_f, osc_sigma2, osc_c, np.reshape(osc_tau2, (1, 1))])
        [osc_mean[0:2*K, :, K-1], osc_cov[0:2*K, 0:2*K, :, K-1]] = osc_smooth(
            Y, fs, osc_param[K-1:K, 0:(2*J+1)*K+1])

        for k in range(K):
            osc_phase[k, :, K-1] = np.arctan2(
                osc_mean[2*k+1, :, K-1], osc_mean[2*k, :, K-1])

    return [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase]
