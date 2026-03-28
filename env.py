import gymnasium as gym

env = gym.make("CartPole-v1")

observation, info = env.reset()

episode_over = False

total_reward = 0
i = 0

while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated
    print(terminated, truncated)


print(f"Episode finished. Total reward: {total_reward}")


import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2):
        # input_dim: dimension of state space
        self.l1 = nn.Linear(input_dim, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, 2)  # mean, std

    def forward(self, x):
        """
        input: state
        output: (mean, std)
        """
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)

        return x


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2):
        # input_dim: dimension of state and action space
        self.l1 = nn.Linear(input_dim, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, 1)  # mean, std

    def forward(self, x):
        """
        input: state
        output: real number
        """
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)

        return x
