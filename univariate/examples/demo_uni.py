import numpy as np
import numpy.linalg as LA
import scipy.io
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from pyoscdecomp.osc_decomp import osc_decomp
from pyoscdecomp.osc_ll_hess import osc_ll_hess
from pyoscdecomp.plot.osc_plot import osc_plot
from pyoscdecomp.plot.osc_phase_plot import osc_phase_plot
from pyoscdecomp.plot.osc_spectrum_plot import osc_spectrum_plot

def demo_uni():
    """
    Function implementing MATLAB script "demo_uni.m"

    Returns:
        osc_AIC: ndarray, shape = (1, MAX_OSC)
            AIC of the oscillator model
        osc_mean: ndarray, shape = (2*MAX_OSC, T, MAX_OSC)
            smoothed coordinate of each oscillator
        osc_cov: ndarray, shape = (2*MAX_OSC, 2*MAX_OSC, T, MAX_OSC)
            smoothed covariance of each oscillator
        osc_phase:ndarray, shape = (MAX_OSC, T, MAX_OSC)
            estimated phase of each oscillation component
        minAIC: float
            min(AIC)
        K: int
            argmin(AIC) + 1
        osc_a: ndarray, shape = (1, K)
            estimated a_1,...,a_K
        osc_f: ndarray, shape = (1, K)
            estimated f_1,...,f_K
        osc_sigma2: ndarray, shape = (1, K)
            estimated sigma_1^2,...,sigma_K^2 
        osc_tau2: float
            estimated tau^2
        hess: ndarray, shape = (3*K+1, 3*K+1)
            Hessian of the negative log-likelihood
        grad: ndarray, shape = (3*K+1, 1)
            gradient of the negative log-likefood
        mll: float
            negative log-likelihood
        cov_est: ndarray, shape = (3*K+1, 3*K+1)
            approximated asymptotic covariance matrix
    """
    # Load 'CanadianLynxData.mat' using scipy.io.loatdmat.
    input_file = parent_dir + '/CanadianLynxData.mat'
    matdata = scipy.io.loadmat(input_file)

    lynx = matdata['lynx']
    y = np.log(lynx)
    y = y - np.mean(y)

    fs = 1.0
    MAX_OSC = 6
    [
        osc_param, 
        osc_AIC, 
        osc_mean, 
        osc_cov,
        osc_phase] = osc_decomp(
            y, fs, MAX_OSC=MAX_OSC)

    minAIC = np.min(osc_AIC)
    K = np.argmin(osc_AIC) + 1

    osc_a = osc_param[K-1:K, 0:K]
    osc_f = osc_param[K-1:K, K:2*K]
    osc_sigma2 = osc_param[K-1:K, 2*K:3*K]
    osc_tau2 = osc_param[K-1:K, 3*K:3*K+1].item()

    hess, grad, mll = osc_ll_hess(y, fs, osc_param[K-1:K, 0:3*K+1])
    cov_est = LA.inv(hess)

    print('The number of oscillators is K={}'.format(K))
    print('The periods of K oscillators are:')
    for k in range(K):
        period = 1/osc_f[0, k]
        low = 1/(osc_f[0, k] + 1.96 * np.sqrt(cov_est[K+k, K+k]))
        up = 1/(osc_f[0, k] - 1.96 * np.sqrt(cov_est[K+k, K+k]))
        print(
            ' {period:.2f} (95%% CI: [{low:.2f} {up:.2f}]) years\n'.format(
                period=period, low=low, up=up) )

    osc_plot(osc_mean, osc_cov, fs, K)
    osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K)
    osc_spectrum_plot(y, fs, osc_a, osc_f, osc_sigma2, osc_tau2)

    return [osc_AIC, osc_mean, osc_cov, osc_phase,
            minAIC, K, osc_a, osc_f, osc_sigma2, osc_tau2,
            hess, grad, mll, cov_est]

if __name__ == '__main__':
    demo_uni()
