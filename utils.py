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


def criterion_actor(log_pi, state_action_value):
    return torch.mean(log_pi - state_action_value)


def criterion_critic(q_out, v_out, reward, alpha=1.0, gamma=0.9):
    scaled_reward = reward / (alpha + EPSILON)

    loss_q = 0.5 * (q_out - scaled_reward - gamma * v_out) ** 2

    return torch.mean(loss_q)


def criterion_value(q_out, v_out, log_pi):
    loss = 0.5 * (v_out - (q_out - log_pi)) ** 2

    return torch.mean(loss)


class ActionTransition:
    def __init__(self, a_min, a_max):
        self.a_mean = (a_min + a_max) / 2
        self.a_scale = a_max - self.a_mean

    def agent2env(self, action):
        return self.a_scale * action + self.a_mean

    def env2agent(self, action):
        return (action - self.a_mean) / (self.a_scale + EPSILON)
