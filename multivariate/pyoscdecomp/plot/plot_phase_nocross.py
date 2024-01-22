import numpy as np

def plot_phase_nocross(t, phase, lineoption: str, linewidth: int, ax):
    """
    Args:
        t: ndarray, shape = (T)
        phase: ndarray, shape = (T)
        lineoption: str
        linewidth: int
        ax: AxesSubplot object
    Returns:
        ret: int
    """    
    for i in range(len(t) - 1):
        if (phase[i]-phase[i + 1] > np.pi):
            turn = (
                t[i] + (np.pi - phase[i]) 
                / (phase[i + 1] + 2*np.pi - phase[i]) 
                * (t[i + 1] - t[i]))
            ax.plot(
                [t[i], turn], [phase[i], np.pi], 
                lineoption, linewidth=linewidth)
            ax.plot(
                [turn, t[i + 1]],[-np.pi, phase[i + 1]], 
                lineoption, linewidth=linewidth)
        else:
            if (phase[i]-phase[i + 1] < -np.pi):
                turn = (
                    t[i] + (phase[i] + np.pi) 
                    / (phase[i] - phase[i + 1] + 2 * np.pi) 
                    * (t[i + 1] - t[i]))
                ax.plot(
                    [t[i], turn], [phase[i], -np.pi], 
                    lineoption, linewidth=linewidth)
                ax.plot(
                    [turn, t[i + 1]], [np.pi, phase[i + 1]], 
                    lineoption, linewidth=linewidth)
            else:
                ax.plot(
                    [t[i], t[i + 1]], [phase[i], phase[i + 1]], 
                    lineoption, linewidth=linewidth)
    ret = 0
    return ret
