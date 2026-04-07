from utils.misc import set_seed, init_csv_log, log_to_csv
import gymnasium as gym
import torch

from model.actor_critic import Actor, Critic
from replay_buffer import ReplayBuffer

from utils.functions import get_space_dim, ActionTransition

import torch.optim as optim
import os


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

    # Define logging path
    experiment_name = f"al{alpha}_lr{lr}_hi{hidden_size}"
    log_dir = f"experiments/{experiment_name}"
    log_path = f"{log_dir}/metrics.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Set Environment
    # env = gym.make(
    #     "MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1
    # )
    # env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

    state_dim = get_space_dim(env.observation_space)
    action_dim = get_space_dim(env.action_space)

    action_transition = ActionTransition(
        env.action_space.low[0], env.action_space.high[0]
    )
    replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

    actor = Actor(state_dim, action_dim, hidden_size)
    critic1 = Critic(state_dim, action_dim, hidden_size)
    critic2 = Critic(state_dim, action_dim, hidden_size)
    target_net1 = Critic(state_dim, action_dim, hidden_size)
    target_net2 = Critic(state_dim, action_dim, hidden_size)
    target_net1.load_state_dict(critic1.state_dict())
    target_net2.load_state_dict(critic2.state_dict())

    # target_h = -action_dim
    # log_alpha = torch.zeros(1, requires_grad=True)

    optimizer_actor = optim.Adam(actor.parameters(), lr=lr)
    optimizer_critic1 = optim.Adam(critic1.parameters(), lr=lr)
    optimizer_critic2 = optim.Adam(critic2.parameters(), lr=lr)
    # optimizer_alpha = optim.Adam([log_alpha], lr=lr)

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

        while not done:
            """
            environment interaction
            """
            with torch.no_grad():
                action, log_pi = actor.rsample(torch.from_numpy(state).view(1, -1))
                action = action.numpy()[0]

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
            # alpha = torch.exp(log_alpha)

            # Batch Sampling
            b_state, b_action, b_reward, b_next_state, b_terminated = (
                replay_buffer.sample(batch_size)
            )

            """
            Update Critic
            """
            q1_out = critic1.forward(b_state, b_action)
            q2_out = critic2.forward(b_state, b_action)

            # Get Target of Critic
            with torch.no_grad():
                next_action, log_pi_next_state = actor.rsample(b_next_state)

                target_q1 = target_net1.forward(b_next_state, next_action)
                target_q2 = target_net2.forward(b_next_state, next_action)
                target = b_reward + gamma * (
                    torch.min(target_q1, target_q2) - alpha * log_pi_next_state
                ) * (1 - b_terminated)

            loss_critic1 = torch.mean(0.5 * (q1_out - target) ** 2)
            loss_critic2 = torch.mean(0.5 * (q2_out - target) ** 2)

            optimizer_critic1.zero_grad()
            optimizer_critic2.zero_grad()
            loss_critic1.backward()
            loss_critic2.backward()
            optimizer_critic1.step()
            optimizer_critic2.step()

            # Update Actor
            curr_action, log_pi_state = actor.rsample(b_state)
            q1_out_c = critic1.forward(b_state, curr_action)
            q2_out_c = critic2.forward(b_state, curr_action)

            q_min = torch.min(q1_out_c, q2_out_c)
            loss_actor = torch.mean(alpha * log_pi_state - q_min)

            optimizer_actor.zero_grad()
            loss_actor.backward()
            optimizer_actor.step()

            # # Update Alpha
            # loss_alpha = -(log_alpha * (log_pi_state + target_h).detach()).mean()
            #
            # optimizer_alpha.zero_grad()
            # loss_alpha.backward()
            # optimizer_alpha.step()

            """
            polyak averaging for target_q
            """

            with torch.no_grad():
                for i, j in zip(critic1.parameters(), target_net1.parameters()):
                    new = tau * i + (1 - tau) * j
                    j.copy_(new)
                for i, j in zip(critic2.parameters(), target_net2.parameters()):
                    new = tau * i + (1 - tau) * j
                    j.copy_(new)
        if (episode + 1) % 10 == 0:
            print(f"------episode {episode + 1}---------")
            print(f"reward: {total_reward}")
            torch.save(actor.state_dict(), os.path.join(log_dir, f"e_{episode + 1}.pt"))

        reward_list.append(total_reward)
    env.close()
    print(f"Episode finished. Total reward: {sum(reward_list) / len(reward_list)}")

    init_csv_log(log_path, ["episode", "reward"])

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
