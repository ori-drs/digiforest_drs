import numpy as np


def l2(r, c=1.0):
    return (r / c) ** 2


def welsh(r, c=1.0):
    return 1.0 - np.exp(-0.5 * (r / c) ** 2)


def smooth_l1(r, c=1.0):
    return np.sqrt((r / c) ** 2 + 1.0) - 1.0


def geman_mcclure(r, c=1.0):
    return 2 * (r / c) ** 2 / ((r / c) ** 2 + 4)
