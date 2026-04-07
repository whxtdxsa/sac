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
    return log_p.view(-1, 1)


class CriterionByRewardScaling:
    def __init__(self, alpha=0.0, gamma=0.99, eps=EPSILON):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def actor(self, log_pi, q):
        loss = log_pi - q

        return torch.mean(loss)

    def critic(self, q, v, reward):
        scaled_reward = reward / (self.alpha + self.eps)
        loss_q = 0.5 * (q - (scaled_reward + self.gamma * v)) ** 2

        return torch.mean(loss_q)

    def value(self, q, v, log_pi):
        loss = 0.5 * (v - (q - log_pi)) ** 2

        return torch.mean(loss)


class Criterion:
    def __init__(self, alpha=0.0, gamma=0.99, eps=EPSILON):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def actor(self, log_pi, q):
        loss = self.alpha * log_pi - q

        return torch.mean(loss)

    def critic(self, q, v, reward):
        loss = 0.5 * (q - (reward + self.gamma * v)) ** 2

        return torch.mean(loss)

    def value(self, q, v, log_pi):
        loss = 0.5 * (v - (q - self.alpha * log_pi)) ** 2

        return torch.mean(loss)


class ActionTransition:
    def __init__(self, a_min, a_max, eps=EPSILON):
        self.a_mean = (a_min + a_max) / 2
        self.a_scale = a_max - self.a_mean
        self.eps = eps

    def agent2env(self, action):
        return self.a_scale * action + self.a_mean

    def env2agent(self, action):
        return (action - self.a_mean) / (self.a_scale + self.eps)
