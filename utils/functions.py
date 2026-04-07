import torch
import math
import gymnasium as gym

EPSILON = 1e-6


def get_space_dim(space):
    if isinstance(space, gym.spaces.Box):
        return space.shape[0]
    elif isinstance(space, gym.spaces.Discrete):
        return space.n


def gaussian_log_prob(x, mean, std, log_std):
    d = x.shape[-1]

    log_p = -0.5 * (
        torch.sum(((x - mean) / (std + EPSILON)) ** 2, dim=-1)
        + 2 * torch.sum(log_std, dim=-1)
        + d * math.log(2 * math.pi)
    )
    return log_p.view(-1, 1)


class ActionTransition:
    def __init__(self, a_min, a_max, eps=EPSILON):
        self.a_mean = (a_min + a_max) / 2
        self.a_scale = a_max - self.a_mean
        self.eps = eps

    def agent2env(self, action):
        return self.a_scale * action + self.a_mean

    def env2agent(self, action):
        return (action - self.a_mean) / (self.a_scale + self.eps)
