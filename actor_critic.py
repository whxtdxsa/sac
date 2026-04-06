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
        log_pi = log_pi_u - torch.sum(torch.log(1 - a**2 + EPSILON), dim=-1)
        return a, log_pi

    def loss(self, state, Q):
        action, log_pi = self.rsample(state)
        loss = log_pi - Q.forward(state, action)
        return loss


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.V = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def loss_q(self, state, action, reward, next_state, done, alpha=1, gamma=0.9):
        with torch.no_grad():
            v_out = self.V(next_state) * (1 - done)

        scaled_reward = reward / (alpha + EPSILON)

        state_action = torch.cat([state, action], dim=-1)
        q1_out = self.Q1(state_action)
        q2_out = self.Q2(state_action)

        loss_q1 = 0.5 * (q1_out - scaled_reward - gamma * v_out) ** 2
        loss_q2 = 0.5 * (q2_out - scaled_reward - gamma * v_out) ** 2

        return torch.mean(loss_q1), torch.mean(loss_q2)

    def loss_v(self, state, actor):
        action, log_prob = actor.rsample(state)
        state_action = torch.cat([state, action], dim=-1)

        with torch.no_grad():
            q_out = torch.min(self.Q1(state_action), self.Q2(state_action))

        loss = 0.5 * (self.V(state) - (q_out - log_prob)) ** 2

        return torch.mean(loss)
