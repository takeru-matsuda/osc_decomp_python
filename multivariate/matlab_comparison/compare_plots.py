import numpy as np
import scipy.io
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from pyoscdecomp.plot.osc_plot import osc_plot
from pyoscdecomp.plot.osc_phase_plot import osc_phase_plot
from pyoscdecomp.plot.osc_spectrum_plot import osc_spectrum_plot


def test_osc_plot():
    """osc_plot のテストです。
       MATLAB の出力値を読み込み、プロットした結果を ./osc_plot.png に出力します。
    """    
    filename = parent_dir + '/matlab_test_outputs/demo_uni.mat'
    data = scipy.io.loadmat(filename)
    osc_mean = np.array(data['osc_mean'])
    osc_cov = np.array(data['osc_cov'])
    fs = np.array(data['fs']).item()
    K = data['K'].item()

    osc_plot(osc_mean, osc_cov, fs, K, save_figure=True)


def test_osc_phase_plot():
    """osc_phase_plot のテストです。
       MATLAB の出力値を読み込み、プロットした結果を 
       ./osc_phase_plot.png に出力します。
    """    
    filename = parent_dir + '/matlab_test_outputs/demo_uni.mat'
    data = scipy.io.loadmat(filename)
    osc_phase = np.array(data['osc_phase'])
    osc_mean = np.array(data['osc_mean'])
    osc_cov = np.array(data['osc_cov'])
    fs = np.array(data['fs']).item()
    K = data['K'].item()

    osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K, save_figure=True)


def test_osc_spectrum_plot():
    """osc_spectrum_plot のテストです。
       MATLAB の出力値を読み込み、プロットした結果を 
       ./osc_spectrum_plot.png に出力します。
    """    
    filename = parent_dir + '/matlab_test_outputs/demo_uni.mat'
    data = scipy.io.loadmat(filename)
    osc_a = np.array(data['osc_a'])
    osc_f = np.array(data['osc_f'])
    osc_sigma2 = np.array(data['osc_sigma2'])
    osc_tau2 = np.array(data['osc_tau2'])
    fs = np.array(data['fs']).item()
    y = np.array(data['y'])

    osc_spectrum_plot(
        y, fs, osc_a, osc_f, osc_sigma2, osc_tau2, save_figure=True)


if __name__ == '__main__':
    test_osc_plot()
    test_osc_phase_plot()
    test_osc_spectrum_plot()




