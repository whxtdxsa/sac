import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(
            hidden_size, 2 * action_dim
        )  # mean(action_dim), std_(action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        out = self.l3(x)

        mu, log_std = torch.split(out, self.action_dim, dim=-1)

        log_std = torch.clamp(log_std, min=-20, max=2)

        return mu, log_std

    def rsample(self, state):
        mu, log_std = self.forward(state)
        sigma = torch.exp(log_std)
        eps = torch.randn_like(log_std)

        a = mu + sigma * eps

        log_pi = -0.5
            * (
                (a - mu).transpose() * (sigma**2).transpose() * (a - mu)
                + torch.log(sigma * sigma.transpose())
                + self.action_dim * torch.log(2 * torch.pi)
            )
        

        return a, log_pi


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        # input_dim: dimension of state and action space
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        out = self.l3(x)

        return out
