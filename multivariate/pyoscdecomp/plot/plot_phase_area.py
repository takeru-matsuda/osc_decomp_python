import numpy as np
import matplotlib.pyplot as plt


def plot_phase_area(t, phase, lphase, uphase, C, ax):
    """
    Args:
        t: ndarray, shape = (T)
        phase: ndarray, shape = (T)
        lphase: ndarray, shape = (T)
        uphase: ndarray, shape = (T)
        C: tuple, shape = (3)
        ax: AxesSubplot object
    Returns:
        ret: int
    """    
    for i in range(len(t)-1):
        lturn = 0
        uturn = 0
        if ((phase[i] - lphase[i]) * (phase[i + 1] - lphase[i + 1]) < 0):
            if ((lphase[i] < phase[i]) and (phase[i] - phase[i + 1] < np.pi)):
                lturn = -1

            if (
                (lphase[i] > phase[i]) 
                and ((phase[i] - phase[i + 1]) > -np.pi)):
                lturn = 1

        else:
            if ((phase[i] - phase[i + 1]) > np.pi):
                lturn = 1

            if ((phase[i] - phase[i + 1]) < -np.pi):
                lturn = -1

        if ((phase[i] - uphase[i]) * (phase[i + 1] - uphase[i + 1]) < 0):
            if (
                (uphase[i] < phase[i]) 
                and (phase[i] - phase[i + 1] < np.pi)):
                uturn = -1
            
            if (
                (uphase[i] > phase[i]) 
                and (phase[i] - phase[i + 1] > -np.pi)):
                uturn = 1
        else:
            if (phase[i] - phase[i + 1] > np.pi):
                uturn = 1

            if (phase[i] - phase[i + 1] < -np.pi):
                uturn = -1

        if (lphase[i] < uphase[i]):
            if (lphase[i + 1] < uphase[i + 1]):
                ld = lphase[i]
                lu = uphase[i]
                rd = lphase[i + 1] + lturn * 2 * np.pi
                ru = uphase[i + 1] + uturn * 2 * np.pi
                ax.fill(
                    np.array([t[i], t[i + 1], t[i + 1], t[i]]).T, 
                    np.array([ld, rd, ru, lu]).T, color=C)
                ld = lphase[i] - lturn * 2 * np.pi
                lu = uphase[i] - uturn * 2 * np.pi
                rd = lphase[i + 1]
                ru = uphase[i + 1]
                ax.fill(
                    np.array([t[i], t[i + 1], t[i + 1], t[i]]).T,
                    np.array([ld, rd, ru, lu]).T, color=C)
            else:
                ld = lphase[i]
                lu = uphase[i]
                rd = lphase[i + 1] + lturn * 2 * np.pi
                ru = uphase[i + 1] + uturn * 2 * np.pi
                ax.fill(
                    np.array([t[i], t[i + 1], t[i + 1], t[i]]).T, 
                    np.array([ld, rd, ru, lu]).T, color=C)
                ld = lphase[i] - (lturn + uturn) * 2 * np.pi
                lu = uphase[i] - (lturn + uturn) * 2 * np.pi
                rd = lphase[i + 1] - uturn * 2 * np.pi
                ru = uphase[i + 1] - lturn * 2 * np.pi
                ax.fill(
                    np.array([t[i], t[i + 1], t[i + 1], t[i]]).T, 
                    np.array([ld, rd, ru, lu]).T, color=C)
        else:
            if (lphase[i + 1] < uphase[i + 1]):
                ld = lphase[i] - lturn * 2 * np.pi
                lu = uphase[i] - uturn * 2 * np.pi
                rd = lphase[i + 1]
                ru = uphase[i + 1]
                ax.fill(
                    np.array([t[i], t[i + 1], t[i + 1], t[i]]).T, 
                    np.array([ld, rd, ru, lu]).T, color=C)
                ld = lphase[i] + uturn * 2 * np.pi
                lu = uphase[i] + lturn * 2 * np.pi
                rd = lphase[i + 1] + (lturn + uturn) * 2 * np.pi
                ru = uphase[i + 1] + (lturn + uturn) * 2 * np.pi
                ax.fill(
                    np.array([t[i], t[i + 1], t[i + 1], t[i]]).T, 
                    np.array([ld, rd, ru, lu]).T, color=C)
            else:
                if (lturn == 0):
                    ld = lphase[i]
                    lu = uphase[i] + 2 * np.pi
                    rd = lphase[i + 1]
                    ru = uphase[i + 1] + 2 * np.pi
                    ax.fill(
                        np.array([t[i], t[i + 1], t[i + 1], t[i]]).T, 
                        np.array([ld, rd, ru, lu]).T, color=C)
                    ld = lphase[i] - 2 * np.pi
                    lu = uphase[i]
                    rd = lphase[i + 1] - 2 * np.pi
                    ru = uphase[i + 1]
                    ax.fill(
                        np.array([t[i], t[i + 1], t[i + 1], t[i]]).T, 
                        np.array([ld, rd, ru, lu]).T, color=C)
                else:
                    ld = lphase[i]
                    lu = uphase[i] + lturn * 2 * np.pi
                    rd = lphase[i + 1] + lturn * 2 * np.pi
                    ru = uphase[i + 1] + lturn * 4 * np.pi
                    ax.fill(
                        np.array([t[i], t[i + 1], t[i + 1], t[i]]).T, 
                        np.array([ld, rd, ru, lu]).T, color=C)
                    ld = lphase[i] - lturn * 2 * np.pi
                    lu = uphase[i]
                    rd = lphase[i + 1]
                    ru = uphase[i + 1] + lturn * 2 * np.pi
                    ax.fill(
                        np.array([t[i], t[i + 1], t[i + 1], t[i]]).T, 
                        np.array([ld, rd, ru, lu]).T, color=C)
                    ld = lphase[i] - lturn * 4 * np.pi
                    lu = uphase[i] - lturn * 2 * np.pi
                    rd = lphase[i + 1] - lturn * 2 * np.pi
                    ru = uphase[i + 1]
                    ax.fill(
                        np.array([t[i], t[i + 1], t[i + 1], t[i]]).T, 
                        np.array([ld, rd, ru, lu]).T, color=C)

    ax.set_ylim([-np.pi, np.pi])
    ret = 0
    return ret
