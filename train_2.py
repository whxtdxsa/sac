import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo


def criterion_actor(log_pi, q_out, alpha=1.0):
    return torch.mean(alpha * log_pi - q_out)


def criterion_critic(q_out, v_out, reward, gamma=0.99):
    loss_q = 0.5 * (q_out - reward - gamma * v_out) ** 2

    return torch.mean(loss_q)


def criterion_value(q_out, v_out, log_pi, alpha=1.0):
    loss = 0.5 * (v_out - (q_out - alpha * log_pi)) ** 2
    return torch.mean(loss)


from actor_critic import Actor, Critic, Value
from replay_buffer import ReplayBuffer

import torch.optim as optim


def get_space_dim(space):
    if isinstance(space, gym.spaces.Box):
        return space.shape[0]
    elif isinstance(space, gym.spaces.Discrete):
        return space.n


env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
# env = RecordVideo(env, video_folder="./pendulum_videos", episode_trigger=lambda x: True)

state_dim = get_space_dim(env.observation_space)
action_space = env.action_space
action_dim = get_space_dim(env.action_space)
replay_buffer = ReplayBuffer(state_dim, action_dim, 10**6)
batch_size = 256
GAMMA = 0.99
ALPHA = 0.2

actor = Actor(state_dim, action_dim, 256)
critic1 = Critic(state_dim, action_dim, 256)
critic2 = Critic(state_dim, action_dim, 256)
value = Value(state_dim, action_dim, 256)
target_value = Value(state_dim, action_dim, 256)

target_value.load_state_dict(value.state_dict())


optimizer_actor = optim.Adam(actor.parameters(), lr=3e-4)
optimizer_critic1 = optim.Adam(critic1.parameters(), lr=3e-4)
optimizer_critic2 = optim.Adam(critic2.parameters(), lr=3e-4)
optimizer_value = optim.Adam(value.parameters(), lr=3e-4)


from tqdm import tqdm
from utils import ActionTransition

action_transition = ActionTransition(action_space.low[0], action_space.high[0])

TAU = 5e-3
EPISODES = 1000
reward_list = []

while len(replay_buffer) < 2000:
    state, info = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        action = action_transition.env2agent(action)
        replay_buffer.insert(state, action, reward, next_state, terminated)
        state = next_state


for episode in range(EPISODES):
    state, info = env.reset()

    done = False
    total_reward = 0.0
    total_q_min = []
    total_lp_out = []
    total_v_out = []

    while not done:
        action, log_pi = actor.rsample(torch.from_numpy(state[None, :]))
        detached_action = action[0].detach().cpu().numpy()

        next_state, reward, terminated, truncated, info = env.step(
            action_transition.agent2env(detached_action)
        )
        done = terminated or truncated
        total_reward += reward
        replay_buffer.insert(state, detached_action, reward, next_state, terminated)
        state = next_state

        b_state, b_action, b_reward, b_next_state, b_terminated = replay_buffer.sample(
            batch_size
        )

        a_out, lp_out = actor.rsample(b_state)
        v_out = value.forward(b_state)
        with torch.no_grad():
            target = target_value.forward(b_next_state) * (1 - b_terminated.float())

        q1_out = critic1.forward(b_state, b_action)
        q2_out = critic2.forward(b_state, b_action)

        optimizer_critic1.zero_grad()
        optimizer_critic2.zero_grad()
        loss_critic1 = criterion_critic(q1_out, target, b_reward, GAMMA)
        loss_critic2 = criterion_critic(q2_out, target, b_reward, GAMMA)
        loss_critic1.backward()
        loss_critic2.backward()
        optimizer_critic1.step()
        optimizer_critic2.step()

        q1_out_c = critic1.forward(b_state, a_out)
        q2_out_c = critic2.forward(b_state, a_out)
        q_min_c = torch.min(q1_out_c, q2_out_c)

        optimizer_value.zero_grad()
        loss_value = criterion_value(q_min_c.detach(), v_out, lp_out.detach(), ALPHA)
        loss_value.backward()
        optimizer_value.step()

        optimizer_actor.zero_grad()
        loss_actor = criterion_actor(lp_out, q_min_c, ALPHA)
        loss_actor.backward()
        optimizer_actor.step()

        with torch.no_grad():
            for i, j in zip(value.parameters(), target_value.parameters()):
                new = TAU * i + (1 - TAU) * j
                j.copy_(new)

        total_q_min.append(q_min_c.mean().item())
        total_lp_out.append(lp_out.mean().item())
        total_v_out.append(v_out.mean().item())

    if (episode + 1) % 10 == 0:
        print(f"------episode {episode + 1}---------")
        print(f"reward: {total_reward}")
        print(
            f"Q-value: {sum(total_q_min) / len(total_q_min):.2f}, Log-Pi: {sum(total_lp_out) / len(total_lp_out):.2f}, V-value: {sum(total_v_out) / len(total_v_out):.2f}"
        )

    reward_list.append(total_reward)
env.close()
print(f"Episode finished. Total reward: {sum(reward_list) / len(reward_list)}")
