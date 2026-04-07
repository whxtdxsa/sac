import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from actor_critic import Actor, Critic


from actor_critic import Actor, Critic
from replay_buffer import ReplayBuffer
import torch


def get_space_dim(space):
    if isinstance(space, gym.spaces.Box):
        return space.shape[0]
    elif isinstance(space, gym.spaces.Discrete):
        return space.n


env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
# env = RecordVideo(env, video_folder="./pendulum_videos", episode_trigger=lambda x: True)

state_dim = get_space_dim(env.observation_space)
action_dim = get_space_dim(env.action_space)
replay_buffer = ReplayBuffer(10**6)
batch_size = 16

actor = Actor(state_dim, action_dim, 256)
critic = Critic(state_dim, action_dim, 256)

state, info = env.reset()

done = False
total_reward = 0.0

while not done:
    action = actor.rsample(state)[0]
    # action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    replay_buffer.insert(state, action, reward, next_state, done)
    batch = torch.split(replay_buffer.sample(batch_size))
    total_reward += reward
    done = terminated or truncated
    replay_buffer.insert(state, action, reward, next_state, done)

env.close()
print(f"Episode finished. Total reward: {total_reward}")

class SAC():
    def __init__()
