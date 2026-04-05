import torch
import math


def gaussian_log_prob(mu, sigma, x):
    log_p = -0.5 * (
        (x - mu) ** 2 / sigma**2 + 2 * torch.log(sigma) + torch.log(2 * math.pi)
    )
