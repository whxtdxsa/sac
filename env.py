import gymnasium as gym
from gymnasium.wrappers import RecordVideo

env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
env = RecordVideo(env, video_folder="./pendulum_videos", episode_trigger=lambda x: True)

observation, info = env.reset()

episode_over = False
total_reward = 0.0


from actor_critic import Actor, Critic

# Actor()

print(env.observation_space)
print(env.action_space)

while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

env.close()
print(f"Episode finished. Total reward: {total_reward}")
