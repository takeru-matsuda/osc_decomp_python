import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Optional


def osc_plot(
    osc_mean, osc_cov, 
    fs: float, K: int, 
    conf: Optional[float] = None,
    save_figure: Optional[bool] = True):
    """
    Args:
        osc_mean: ndarray, shape = (2*MAX_OSC, T, MAX_OSC)
        osc_cov: ndarray, shape = (2*MAX_OSC, 2*MAX_OSC, T, MAX_OSC)
        fs: float
        K: int
        conf: float. Defaults to None.
        save_figure: bool. Defaults to True.
                Plotted figure is saved to "./osc_plot.png" 
                if save_figure is True.
    """    
    if (conf is None):
        conf = 2 * stats.norm.cdf(1) - 1
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    T = osc_mean.shape[1]

    for k in range(K):
        ax = fig.add_subplot(K, 1, k+1)
        fig.show()
        std_bar = (np.sqrt(osc_cov[2*k, 2*k, 0:T, K-1])).reshape(T)
        xx = np.concatenate([np.arange(1, T+1)/fs, np.arange(T, 0, -1)/fs])
        yy = np.concatenate([
            osc_mean[2*k, 0:T, K-1] 
            - stats.norm.ppf(conf+(1-conf)/2) * std_bar, 
            osc_mean[2*k,np.arange(T-1, -1, -1),K-1] 
            + stats.norm.ppf(conf+(1-conf)/2) 
            * std_bar[np.arange(T-1, -1, -1)]])
        ax.fill(xx, yy, color=(.8, .8, .8))
        ax.plot(np.arange(1, T+1)/fs, osc_mean[2*k, 0:T, K-1], 'k-')
        ax.set_xlim([1.0/fs,  T/fs])

        ax.xaxis.set_major_locator(
            ticker.FixedLocator(ax.get_xticks().tolist()))
        ax.yaxis.set_major_locator(
            ticker.FixedLocator(ax.get_yticks().tolist()))
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        ax.set_xticklabels(xticklabels, fontsize=12)
        ax.set_yticklabels(yticklabels, fontsize=12)

    if save_figure:
        fig.savefig('osc_plot.png')
        fig.show()
