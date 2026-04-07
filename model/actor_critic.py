import torch
import torch.nn as nn
from utils.functions import gaussian_log_prob

EPSILON = 1e-6


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        """
        assume actions are linearly independent
        output: mean(action_dim), std(action_dim) 
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.PI = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * action_dim),
        )

    def forward(self, state):
        out = self.PI(state)

        mu, log_sigma = torch.split(out, self.action_dim, dim=-1)
        log_sigma = torch.clamp(log_sigma, min=-20, max=2)

        return mu, log_sigma

    def rsample(self, state):
        """
        reparameterization trick
        """
        mu, log_sigma = self.forward(state)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)

        u = mu + sigma * eps
        a = torch.tanh(u)
        log_pi_u = gaussian_log_prob(u, mu, sigma, log_sigma)
        log_pi = log_pi_u - torch.sum(torch.log(1 - a**2 + EPSILON), dim=-1).view(-1, 1)

        return a, log_pi


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.Q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.Q(state_action)
