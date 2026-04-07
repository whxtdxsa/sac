import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import gaussian_log_prob

EPSILON = 1e-6


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        """
        assume actions are linearly independent
        output: mean(action_dim), std_(action_dim) 
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 2 * action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        out = self.l3(x)

        mu, log_sigma = torch.split(out, self.action_dim, dim=-1)

        log_sigma = torch.clamp(log_sigma, min=-20, max=2)

        return mu, log_sigma

    def rsample(self, state):
        mu, log_sigma = self.forward(state)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)

        # Scaling in
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


class Value(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.state_dim = state_dim

        self.V = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.V(state)
