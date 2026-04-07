import torch

EPSILON = 1e-6


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
