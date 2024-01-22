import numpy as np
from scipy.linalg import sqrtm
from math import ceil
import numpy.linalg as LA


def angle_conf_MC(mu, Sigma, prob: float, seeds):
    """
    Args:
        mu: ndarray, shape = (2)
        Sigma: ndarray, shape = (2, 2)
        prob: float
        seeds: ndarray, shape = (2, nmc)

    Returns:
        conf: ndarray, shape = (2)
    """    
    nmc = seeds.shape[1]
    phi = np.arctan2(mu[1], mu[0])
    P = np.array([
        [np.cos(phi), np.sin(phi)], 
        [-np.sin(phi), np.cos(phi)]])
    Sigmasqrt = sqrtm(Sigma)
    seeds = (
        LA.norm(mu) * np.block([
            [np.ones((1, nmc))], 
            [np.zeros((1, nmc))]]) 
        + P @ Sigmasqrt @ seeds)
    phases = np.arctan2(seeds[1, :], seeds[0, :])
    I = np.argsort(np.abs(phases))
    phases = np.sort(phases[I[0:ceil(prob*nmc)]])
    conf = np.array([phi + phases[0], phi + phases[-1]])
    if (conf[0] < -np.pi):
        conf[0] += 2 * np.pi

    if (conf[1] > np.pi):
        conf[1] -= 2 * np.pi

    return conf



