import torch
import math


EPSILON = 1e-6


def gaussian_log_prob(x, mean, std, log_std):
    d = x.shape[-1]

    log_p = -0.5 * (
        torch.sum(((x - mean) / (std + EPSILON)) ** 2, dim=-1)
        + 2 * torch.sum(log_std, dim=-1)
        + d * math.log(2 * math.pi)
    )
    return log_p
