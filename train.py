import argparse
import os
import numpy as np
import torch
import DDPG
import utils
import pandas as pd
import environment
import gen_channel as gen
import time
from tqdm import trange

def whiten(state):
    return (state - np.mean(state)) / np.std(state)

def main():
    start_time = time.perf_counter()

    # Training-specific parameters

    gpu = 0                # "gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)
    # buffer_size = 10
    buffer_size = 100000     # "buffer_size", default=100000, type=int, help='Size of the experience replay buffer (default: 100000)
    batch_size = 32
    # batch_size = 16          # "batch_size", default=16, help='Batch size (default: 16)
    seed = 0
                            # "save_model", action="store_true", help='Save model and optimizer parameters'
                            # "load_model", default="", help='Model load file name; if empty, does not load'

    # Environment-specific parameters

    num_antennas = 2            # "num_antennas", default=4, type=int, help='Number of antennas in the BS'
    num_RIS_elements = 16          # "num_RIS_elements", default=4, type=int, help='Number of RIS elements'
    num_users = 2                  # "num_users", default=4, type=int, help='Number of users'
    # num_users = 4    
    power_t = 10                    # "power_t", default=0, type=float, help='Transmission power for the constrained optimization in dBm'
    # num_time_steps_per_eps = 10
    num_time_steps_per_eps = 10000   # "num_time_steps_per_eps", default=20000, type=int, help='Maximum number of steps per episode (default: 20000)
    num_eps = 10000                   # "num_eps", default=10, type=int, help='Maximum number of episodes (default: 5000)
    awgn_var = -169              # "awgn_var", default=-169, type=float, help='The noise power spectrum density in dBm/Hz (default: -169)
    BW = 240000                  # "BW", default=240000, type=int, help='the transmission bandwidth in Hz (default: 240k)

    # Algorithm-specific parameters

    discount = 0.99                # "discount", default=0.99, help='Discount factor for reward (default: 0.99)'
    tau = 1e-3                     # "tau", default=1e-3, type=float, help='Learning rate in soft/hard updates of the target networks (default: 0.001)'
    lr = 1e-3                      # "lr", default=1e-3, type=float, help='Learning rate for the networks (default: 0.001)'
    decay = 1e-5                   # "decay", default=1e-5, type=float, help='Decay rate for the networks (default: 0.00001)'

    file_name = f"{num_antennas}_{num_RIS_elements}_{num_users}_{power_t}_{num_eps}"

    env = environment.RIS_MISO(num_antennas, num_RIS_elements, num_users, AWGN_var=awgn_var)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "power_t": power_t,
        "max_action": max_action,
        "M": num_antennas,
        "N": num_RIS_elements,
        "K": num_users,
        "actor_lr": lr,
        "critic_lr": lr,
        "actor_decay": decay,
        "critic_decay": decay,
        "device": device,
        "discount": discount,
        "tau": tau
    }

    agent = DDPG.DDPG(**kwargs)
    replay_buffer = utils.ExperienceReplayBuffer(state_dim, action_dim, max_size=buffer_size)
    instant_rewards = []
    eps_max_reward = []
    episode_num = 0
    predict_mode = False

    channel_file = f'channel_csv/{file_name}.csv'
    # df = pd.read_csv(channel_file)
    df0 = pd.read_csv(channel_file)
    start_at = 0
    df = df0.iloc[start_at:].copy()
    G = df['G'].values
    H_r_all = df['H_r_all'].values
    H_d_all = df['H_d_all'].values

    episode_num = episode_num + start_at

    for eps in trange(int(num_eps)):
        # print('R:',type(H_r_all[eps]))
        # print('D:',type(H_d_all[eps]))
        # print('G:',type(G[eps]))
        H_r_ = string_to_ndarray(H_r_all[eps])
        G_ = string_to_ndarray(G[eps])
        H_d_ = string_to_ndarray(H_d_all[eps])
        # break
        episode_num
        state, done = env.reset(predict_mode, episode_num, G_, H_r_, H_d_), False
        episode_reward = 0
        episode_time_steps = 0
        max_reward = 0
        episonde_max = 0
        state = whiten(state)
        eps_rewards = []
        # print(G)
        # break

        for t in trange(int(num_time_steps_per_eps)):
            action = agent.select_action(np.array(state))

            next_state, reward, done, _ = env.step(action)
            done = 1.0 if t == num_time_steps_per_eps - 1 else float(done)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            state = whiten(state)

            if reward > max_reward:
                max_reward = reward
                episonde_max = episode_time_steps

            agent.update_parameters(replay_buffer, batch_size)
            eps_rewards.append(reward)
            episode_time_steps += 1

            if done:
                print(f"\nTotal T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episonde_max} Max. Reward: {max_reward:.3f}\n")
                episode_reward = 0
                episode_time_steps = 0
                episode_num += 1
                state = whiten(state)
                eps_max_reward.append(max_reward)
                instant_rewards.append(eps_rewards)

                if not episode_num % 200:
                    np.savetxt(f'./train/{file_name}/train/{file_name}_{episode_num}_instant_rewards.csv', instant_rewards, delimiter=';', fmt="%.3f")
                    np.savetxt(f'./train/{file_name}/train/{file_name}_{episode_num}_max_reward.csv', eps_max_reward, delimiter=';', fmt="%.3f")
                    agent.save(f"./train/{file_name}/Model_save/{file_name}__episode_{episode_num}")

    end_time = time.perf_counter()
    n = end_time - start_time
    time_format = time.strftime("%H:%M:%S", time.gmtime(n))
    print("Time in preferred format :-", time_format)

def string_to_ndarray(matrix_str):
    matrix_str=matrix_str.replace('j ', 'j, ').replace("\n", "").replace("] [", "],[").replace("][", "],[")
    matrix_np= np.array(eval(matrix_str))
    return matrix_np

if __name__ == "__main__":
    main()
