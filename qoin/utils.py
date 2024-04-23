import math

import numpy as np
import torch


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = torch.abs(error) <= clip_delta
    squared_loss = .5 * torch.square(cond)
    quadratic_loss = (.5 * (clip_delta ** 2)) + (clip_delta * (torch.abs(error) - clip_delta))
    return torch.mean(torch.where(cond, squared_loss, quadratic_loss))


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except:
        raise ("Error in sigmoid:", x)


def get_state(data, t, n_timesteps):
    """Returns state representation of timesteps
    """
    d = t - n_timesteps + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
    res = []
    for i in range(n_timesteps - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])
