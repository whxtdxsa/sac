import gymnasium as gym

env = gym.make("CartPole-v1")

observation, info = env.reset()

episode_over = False

total_reward = 0
i = 0


from actor_critic import Actor, Critic


# Actor()

print(len(observation))
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated
    print(terminated, truncated)


print(f"Episode finished. Total reward: {total_reward}")
