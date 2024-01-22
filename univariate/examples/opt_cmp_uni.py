import numpy as np
import scipy.io
import time
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from pyoscdecomp.osc_decomp import osc_decomp

def opt_cmp_uni():
    """
    Function implementing MATLAB script "opt_cmp_uni.m"
    
    Returns:
        osc_param1: ndarray, shape = (MAX_OSC, 3*MAX_OSC + 1)
            estimated parameters (quasi-Newton, gradient = numerical difference)
        osc_AIC1: ndarray, shape = (1, MAX_OSC)
            AIC (quasi-Newton, gradient = numerical difference)
        osc_param2: ndarray, shape = (MAX_OSC, 3*MAX_OSC + 1)
            estimated parameters (quasi-Newton, gradient = Kalman filter)
        osc_AIC2: ndarray, shape = (1, MAX_OSC)
            AIC (quasi-Newton, gradient = Kalman filter)
        osc_param3: ndarray, shape = (MAX_OSC, 3*MAX_OSC + 1)
            estimated parameters (trust-region, gradient = Kalman filter)
        osc_AIC3: ndarray, shape = (1, MAX_OSC)
            AIC (trust-region, gradient = Kalman filter)
    """    
    # Load 'CanadianLynxData.mat' using scipy.io.loatdmat.
    input_file = parent_dir + '/CanadianLynxData.mat'
    matdata = scipy.io.loadmat(input_file)

    lynx = matdata['lynx']
    y = np.log(lynx)
    y = y - np.mean(y)

    fs = 1.0
    MAX_OSC = 6
    MAX_AR = 20

    tic = time.time()
    [
        osc_param1, 
        osc_AIC1, 
        osc_mean, 
        osc_cov,
        osc_phase] = osc_decomp(
            y, fs, MAX_OSC=MAX_OSC, MAX_AR=MAX_AR, 
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
            y, fs, MAX_OSC=MAX_OSC, MAX_AR=MAX_AR, 
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
            y, fs, MAX_OSC=MAX_OSC, MAX_AR=MAX_AR,
            algorithm='trust-region', grad=True)
    toc = time.time()
    time3 = toc - tic
    print('time3 = ', time3, '[s]')

    return [osc_param1, osc_AIC1,
            osc_param2, osc_AIC2, 
            osc_param3, osc_AIC3]


if __name__ == '__main__':
    opt_cmp_uni()
