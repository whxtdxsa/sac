from utils.misc import set_seed
import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from model.actor_critic import Actor

from utils.functions import get_space_dim, ActionTransition

import os

seed = 42
alpha = 0.2
lr = 0.0003
hidden_size = 256
epsiode = 120

# Set Seed
set_seed(seed)
# Set Environment
# env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1)
env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
env = RecordVideo(
    env, video_folder="./experiments/pendulum_videos", episode_trigger=lambda x: True
)
# Define logging path
experiment_name = f"al{alpha}_lr{lr}_hi{hidden_size}"
log_dir = f"experiments/{experiment_name}"
weight_file = os.path.join(log_dir, f"e_{epsiode}.pt")

state_dim = get_space_dim(env.observation_space)
action_dim = get_space_dim(env.action_space)
actor = Actor(state_dim, action_dim, hidden_size)
actor.load_state_dict(torch.load(weight_file))

action_transition = ActionTransition(env.action_space.low[0], env.action_space.high[0])


state, _ = env.reset()
done = False
total_reward = 0.0

while not done:
    with torch.no_grad():
        action, log_pi = actor.rsample(torch.from_numpy(state).view(1, -1))
        action = action.detach().cpu().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(
            action_transition.agent2env(action)
        )
        done = terminated or truncated
        total_reward += float(reward)
        state = next_state

env.close()
print(total_reward)
