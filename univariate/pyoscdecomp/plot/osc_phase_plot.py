import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Optional
from numpy.random import randn
from .angle_conf_MC import angle_conf_MC
from .plot_phase_area import plot_phase_area
from .plot_phase_nocross import plot_phase_nocross


def osc_phase_plot(
    osc_phase, osc_mean, osc_cov, 
    fs: float, K: int, 
    conf: Optional[float] = None, 
    nmc: Optional[int] = 1000,
    save_figure: Optional[bool] = True):
    """
    Args:
        osc_phase: _ndarray, shape = (MAX_OSC, T, MAX_OSC)
        osc_mean: ndarray, shape = (2*MAX_OSC, T, MAX_OSC)
        osc_cov: ndarray, shape = (2*MAX_OSC, 2*MAX_OSC, T, MAX_OSC)
        fs: float
        K: int
        conf: float. Defaults to None.
        nmc: int. Defaults to 1000.
        save_figure: bool. Defaults to True.
                     Plotted figure is saved to "./osc_phase_plot.png" 
                     if save_figure is True.
    """    
    if (conf is None):
        conf = 2 * stats.norm.cdf(1) - 1

    T = osc_mean.shape[1]
    phase1 = np.zeros(T)
    phase2 = np.zeros(T)
    seeds = randn(2, nmc)
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    for k in range(K):
        for t in range(T):
            tmp = angle_conf_MC(
                osc_mean[2*k:2*k+2, t, K-1], 
                osc_cov[2*k:2*k+2, 2*k:2*k+2, t, K-1], 
                conf, seeds)
            phase1[t] = tmp[0]
            phase2[t] = tmp[1]

        ax = fig.add_subplot(K, 1, k+1)
        fig.show()
        plot_phase_area(
            np.arange(1, T+1)/fs, osc_phase[k, 0:T, K-1], 
            phase1, phase2, (.8, .8, .8), ax)
        plot_phase_nocross(
            np.arange(1, T+1)/fs, osc_phase[k, 0:T, K-1], 'k-', 2, ax)
        
        ax.set_xlim([1.0/fs, T/fs])
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels([-3.14, 0.0, 3.14], fontsize=12)
    fig.show()

    if (save_figure):
        fig.savefig('osc_phase_plot.png')
        fig.show()
