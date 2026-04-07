from utils import gaussian_log_prob
import torch
import numpy as np


def test_gaussian():
    x = torch.tensor([[2, 4, 6, 8], [1, 3, 5, 6]], dtype=torch.float)
    x = torch.tanh(x)
    mean = torch.rand_like(x)
    std = torch.rand_like(x)
    log_std = torch.log(std)

    log_pi_u = gaussian_log_prob(x, mean, std, log_std)
    log_pi = log_pi_u - torch.sum(torch.log(abs(1 - x**2)), dim=-1)
    print(torch.log(1 - x**2))
    print(log_pi)


import torch.nn as nn


def test_params():
    state_dim = 2
    hidden_size = 256

    V = nn.Sequential(
        nn.Linear(state_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
    )

    params = V.parameters()
    for param in params:
        print(param)


test_params()
