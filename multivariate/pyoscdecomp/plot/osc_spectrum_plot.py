import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Optional
from math import ceil

def osc_spectrum_plot(
    Y, fs: float, 
    osc_a, osc_f, 
    osc_sigma2, 
    osc_tau2: float, 
    osc_c=None,
    save_figure: Optional[bool] = True):
    """
    Args:
        Y: ndarray, shape = (J, T)
        fs: float
        osc_a: ndarray, shape = (1, K)
        osc_f: ndarray, shape = (1, K)
        osc_sigma2: ndarray, shape = (1, K)
        osc_tau2: float
        osc_c: ndarray. Defaults to None.
        save_figure: bool. Defaults to True.
                  Plotted figure is saved to "osc_spectrum_plot_[dim].png"
                  if save_figure is True.
    """    
    K = osc_a.shape[1]
    J = Y.shape[0]
    if (J == 1):
        osc_c = np.zeros((1, 2*K))
        osc_c[0, np.arange(0, 2*K, 2)] = 1
    T = Y.shape[1]
    H = np.zeros((J, 2*K))
    H[0, np.arange(0, 2*K, 2)] = 1
    kk = 0
    for k in range(K):
        for j in range(1, J):
            H[j, 2*k:2*k+2] = osc_c[0, kk:kk+2]
            kk += 2
    for j in range(J):
        peri = np.zeros((1, ceil(T/2))).astype(complex)
        spect = np.zeros((K, ceil(T/2))).astype(complex)
        for i in range(ceil(T/2)):
            peri[0, i] = np.abs(
                Y[j, :] 
                @ np.conjugate(np.exp(
                    -1j * .0 * np.pi * (i + 1) / T * np.arange(1, T+1)
                    )).T)**2 / 2 / np.pi / T
            peri[0, i] = np.abs(
                np.dot(Y[j, :], 
                       np.exp(-1j * 2 * np.pi * (i + 1) 
                       / T * np.arange(1, T+1)))**2 / 2 / np.pi / T)
            for k in range(K):
                a = osc_a[0, k]
                theta = 2 * np.pi * osc_f[0, k] / fs
                A = (
                    (1 - 2 * a**2 * np.cos(theta)**2 + a**4 * np.cos(2*theta)) 
                    / a / (a**2 - 1) / np.cos(theta))
                b = (
                    (A - 2 * a * np.cos(theta) 
                     + np.sqrt((A - 2 * a * np.cos(theta))**2 - 4)) / 2)
                spect[k, i] = (
                    -LA.norm(H[j, 2*k:2*k+2])**2 
                    * osc_sigma2[0, k] * a * np.cos(theta) / b 
                    * np.abs(1 + b * np.exp(-1j * 2 * np.pi * (i+1) / T))**2 
                    / np.abs(
                        1 - 2*a*np.cos(theta)*np.exp(-1j*2*np.pi*(i+1)/T) 
                        + a**2 * np.exp(-1j * 4 * np.pi * (i+1) / T))**2 
                    / 2 / np.pi)
        peri = peri.real
        spect = spect.real
        noise = osc_tau2 / 2 / np.pi * np.ones((1, ceil(T/2)))
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.2)
        ax = fig.add_subplot(1, 1, 1)
        if (J == 1):
            p1 = ax.plot(
                np.arange(1, ceil(T/2)+1)/T*fs, np.log10(peri[0, :]), 
                'k--', label='periodogram')
        else:
            p1 = ax.plot(
                np.arange(1, ceil(T/2)+1)/T*fs, np.log10(peri[0, :]), 
                'k--', label='periodogram of y{}'.format(j+1))
        p2 = ax.plot(
            np.arange(1, ceil(T/2)+1)/T*fs, np.log10(spect[0,:]),
            'r-', label='oscillator')
        for k in range(K):
            ax.plot(
                np.arange(1, ceil(T/2)+1)/T*fs, np.log10(spect[k, :]), 'r-')
        p3 = ax.plot(
            np.arange(1, ceil(T/2)+1)/T*fs, np.log10(noise[0, :]),
            'g+-', label='noise')
        p4 = ax.plot(
            np.arange(1, ceil(T/2)+1)/T*fs, 
            np.log10(np.sum(spect, axis=0)+noise[0, :]), 
            'b*-', label='sum of oscillator & noise')

        ax.set_xlabel('frequency', fontsize=20)
        ax.set_xlim([0, fs/2])
        ax.set_ylabel('log10 power', fontsize=20)
        ax.xaxis.set_major_locator(
            ticker.FixedLocator(ax.get_xticks().tolist()))
        ax.yaxis.set_major_locator(
            ticker.FixedLocator(ax.get_yticks().tolist()))
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        ax.set_xticklabels(xticklabels, fontsize=16)
        ax.set_yticklabels(yticklabels, fontsize=16)
        ax.legend()
        fig.show()

        if save_figure:
            file_name = 'osc_spectrum_plot_{}.png'.format(j+1)
            fig.savefig(file_name)
            fig.show()
