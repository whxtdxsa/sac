from utils.misc import set_seed, init_csv_log, log_to_csv
import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from model.actor_critic import Actor, Critic, Value
from replay_buffer import ReplayBuffer

from utils.functions import get_space_dim, ActionTransition

import importlib
import torch.optim as optim


def run(cfg):
    seed, episodes = (
        cfg["seed"],
        cfg["episodes"],
    )
    batch_size, buffer_size, initial_samples = (
        cfg["batch_size"],
        int(cfg["buffer_size"]),
        cfg["initial_samples"],
    )
    criterion_name = cfg["criterion_name"]
    hidden_size, gamma, alpha, tau, lr = (
        cfg["hidden_size"],
        cfg["gamma"],
        cfg["alpha"],
        cfg["tau"],
        cfg["lr"],
    )

    # Set Seed
    set_seed(seed)

    # Set Criterion
    module = importlib.import_module("criterion")
    Criterion = getattr(module, criterion_name)

    # Set Environment
    env = gym.make(
        "MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1
    )
    # env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
    # env = RecordVideo(env, video_folder="./experiments/pendulum_videos", episode_trigger=lambda x: True)

    state_dim = get_space_dim(env.observation_space)
    action_dim = get_space_dim(env.action_space)

    action_transition = ActionTransition(
        env.action_space.low[0], env.action_space.high[0]
    )
    replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

    actor = Actor(state_dim, action_dim, hidden_size)
    critic1 = Critic(state_dim, action_dim, hidden_size)
    critic2 = Critic(state_dim, action_dim, hidden_size)
    value = Value(state_dim, hidden_size)
    target_value = Value(state_dim, hidden_size)
    target_value.load_state_dict(value.state_dict())

    # criterion = Criterion(ALPHA, GAMMA)
    criterion = Criterion(alpha, gamma)

    optimizer_actor = optim.Adam(actor.parameters(), lr=lr)
    optimizer_critic1 = optim.Adam(critic1.parameters(), lr=lr)
    optimizer_critic2 = optim.Adam(critic2.parameters(), lr=lr)
    optimizer_value = optim.Adam(value.parameters(), lr=lr)

    """
    make initial experience for empty replay_buffer
    """
    while len(replay_buffer) < initial_samples:
        state, _ = env.reset()

        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            action = action_transition.env2agent(action)
            replay_buffer.insert(state, action, reward, next_state, terminated)
            state = next_state
            if len(replay_buffer) >= initial_samples:
                break

    """
    sac
    """
    reward_list = []
    for episode in range(episodes):
        state, _ = env.reset()

        done = False
        total_reward = 0.0
        total_q_min = []
        total_lp_out = []
        total_v_out = []

        while not done:
            """
            environment interaction
            """
            action, log_pi = actor.rsample(torch.from_numpy(state).view(1, -1))
            action = action.detach().cpu().numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(
                action_transition.agent2env(action)
            )
            done = terminated or truncated
            total_reward += float(reward)
            replay_buffer.insert(state, action, reward, next_state, terminated)
            state = next_state

            """
            neural net update
            """
            b_state, b_action, b_reward, b_next_state, b_terminated = (
                replay_buffer.sample(batch_size)
            )

            a_out, lp_out = actor.rsample(b_state)
            v_out = value.forward(b_state)

            q1_out = critic1.forward(b_state, b_action)
            q2_out = critic2.forward(b_state, b_action)

            with torch.no_grad():
                target = target_value.forward(b_next_state) * (1 - b_terminated.float())

            optimizer_critic1.zero_grad()
            optimizer_critic2.zero_grad()
            loss_critic1 = criterion.critic(q1_out, target, b_reward)
            loss_critic2 = criterion.critic(q2_out, target, b_reward)
            loss_critic1.backward()
            loss_critic2.backward()
            optimizer_critic1.step()
            optimizer_critic2.step()

            q1_out_c = critic1.forward(b_state, a_out)
            q2_out_c = critic2.forward(b_state, a_out)
            q_min_c = torch.min(q1_out_c, q2_out_c)

            optimizer_value.zero_grad()
            loss_value = criterion.value(q_min_c.detach(), v_out, lp_out.detach())
            loss_value.backward()
            optimizer_value.step()

            optimizer_actor.zero_grad()
            loss_actor = criterion.actor(lp_out, q_min_c)
            loss_actor.backward()
            optimizer_actor.step()

            """
            polyak averaging for target_v
            """

            with torch.no_grad():
                for i, j in zip(value.parameters(), target_value.parameters()):
                    new = tau * i + (1 - tau) * j
                    j.copy_(new)

            total_q_min.append(q_min_c.mean().item())
            total_lp_out.append(lp_out.mean().item())
            total_v_out.append(v_out.mean().item())

        if (episode + 1) % 40 == 0:
            print(f"------episode {episode + 1}---------")
            print(f"reward: {total_reward}")
            print(
                f"Q-value: {sum(total_q_min) / len(total_q_min):.2f}, Log-Pi: {sum(total_lp_out) / len(total_lp_out):.2f}, V-value: {sum(total_v_out) / len(total_v_out):.2f}"
            )

        reward_list.append(total_reward)
    env.close()
    print(f"Episode finished. Total reward: {sum(reward_list) / len(reward_list)}")

    experiment_name = f"al{alpha}_cr{criterion_name}_lr{lr}_hi{hidden_size}"
    log_path = f"experiments/{experiment_name}/metrics.csv"
    init_csv_log(log_path, ["epsiode", "reward"])

    # Logging
    for episode in range(len(reward_list)):
        log_to_csv(
            log_path,
            {
                "episode": episode,
                "reward": reward_list[episode],
            },
        )


if __name__ == "__main__":
    # Get hyperparams
    import yaml

    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    run(cfg)
