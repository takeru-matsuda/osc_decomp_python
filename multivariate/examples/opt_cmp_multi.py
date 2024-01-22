import numpy as np
import scipy.io
import time
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from pyoscdecomp.osc_decomp import osc_decomp

def opt_cmp_multi():
    """
    Function implementing MATLAB script "opt_cmp_multi.m"
    
    Returns:
        osc_param1: ndarray, shape = (MAX_OSC, (2*J+1)*MAX_OSC + 1)
            estimated parameters (quasi-Newton, gradient = numerical differentiation)
        osc_AIC1: ndarray, shape = (1, MAX_OSC)
            AIC (quasi-Newton, gradient = numerical differentiation)
        osc_param2: ndarray, shape = (MAX_OSC, (2*J+1)*MAX_OSC + 1)
            estimated parameters (quasi-Newton, gradient = Kalman filter)
        osc_AIC2: ndarray, shape = (1, MAX_OSC)
            AIC (quasi-Newton, gradient = Kalman filter)
        osc_param3: ndarray, shape = (MAX_OSC, (2*J+1)*MAX_OSC + 1)
            estimated parameters (trust-region, gradient = Kalman filter)
        osc_AIC3: ndarray, shape = (1, MAX_OSC)
            AIC (trust-region, gradient = Kalman filter)
        osc_param4: ndarray, shape = (MAX_OSC, (2*J+1)*MAX_OSC + 1)
            estimated parameters (trust-region-ncg, gradient = Kalman filter)
        osc_AIC4: ndarray, shape = (1, MAX_OSC)
            AIC (trust-region-ncg, gradient = Kalman filter)
    """    
    filename = parent_dir + '/NorthSouthSunspotData.mat'
    matdata = scipy.io.loadmat(filename)
    dat = matdata['dat']
    Y = np.log(dat + 1)
    J = Y.shape[0]
    T = Y.shape[1]
    Y = Y - np.mean(Y, axis=1).reshape(J, 1) @ np.ones((1, T))
    fs = 12
    MAX_OSC = 6
    MAX_VAR = 20

    tic = time.time()
    [
        osc_param1, 
        osc_AIC1, 
        osc_mean, 
        osc_cov,
        osc_phase] = osc_decomp(
            Y, fs, MAX_OSC=MAX_OSC, MAX_AR=MAX_VAR, 
            algorithm='quasi-newton', grad=False)
    toc = time.time()
    time1 = toc - tic
    print('time1 = ', time1, '[s]')
    
    tic = time.time()
    [
        osc_param2, 
        osc_AIC2, 
        osc_mean, 
        osc_cov,
        osc_phase] = osc_decomp(
            Y, fs, MAX_OSC=MAX_OSC, MAX_AR=MAX_VAR, 
            algorithm='quasi-newton', grad=True)
    toc = time.time()
    time2 = toc - tic
    print('time2 = ', time2, '[s]')

    tic = time.time()
    [
        osc_param3, 
        osc_AIC3, 
        osc_mean,
        osc_cov,
        osc_phase] = osc_decomp(
            Y, fs, MAX_OSC=MAX_OSC, MAX_AR=MAX_VAR,
            algorithm='trust-region', grad=True)
    toc = time.time()
    time3 = toc - tic
    print('time3 = ', time3, '[s]')

    tic = time.time()
    [
        osc_param4, 
        osc_AIC4, 
        osc_mean,
        osc_cov,
        osc_phase] = osc_decomp(
            Y, fs, MAX_OSC=MAX_OSC, MAX_AR=MAX_VAR,
            algorithm='trust-region-ncg', grad=True)
    toc = time.time()
    time4 = toc - tic
    print('time4 = ', time4, '[s]')

    return [osc_param1, osc_AIC1,
            osc_param2, osc_AIC2, 
            osc_param3, osc_AIC3, 
            osc_param4, osc_AIC4]
                

if __name__ == '__main__':
    opt_cmp_multi()



