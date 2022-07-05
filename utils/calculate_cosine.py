import numpy as np

def cosine_angle(x1, x2):
    """
    :param x1:  N, 3
    :param x2:  N, 3
    :return:  N
    """
    up = np.sum(x1 * x2)
    x1_norm = np.linalg.norm(x1, ord=2)
    x2_norm = np.linalg.norm(x2, ord=2)
    down = x1_norm * x2_norm + 1e-30
    res = up / down
    return res