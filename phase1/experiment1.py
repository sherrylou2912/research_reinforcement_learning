import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent import SAC, CQL_SAC
from rl_utils import ReplayBuffer, train_off_policy_agent, train_offline_agent
import multiprocessing as mp
import os

# experiment parameters 
num_trails = 5
num_episodes = 100
num_epochs = 100
num_train_per_epoch = 250
batch_size = 128
minimal_size = 1000
# hyperparameters
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
hidden_dim = 512
gamma = 0.99
tau = 0.005
beta = 7.0
num_random = 10


def make_env():
    return gym.make("Hopper-v3")

def plot_with_std(data, label, color):
    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    x = np.arange(len(mean))
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

def run_trial(trial_id, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)
    print(f"[Trial {trial_id}] Running on GPU {gpu_id}")

    env_online = make_env()
    env_offline1 = make_env()
    env_offline2 = make_env()

    state_dim = env_online.observation_space.shape[0]
    action_dim = env_online.action_space.shape[0]
    action_bounds = env_online.action_space.high[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_entropy = - action_dim

    online_agent = SAC(state_dim, hidden_dim, action_dim, action_bounds, 
                       actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)
    offline_agent1 = SAC(state_dim, hidden_dim, action_dim, action_bounds, 
                         actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)
    offline_agent2 = CQL_SAC(state_dim, hidden_dim, action_dim, action_bounds,
                             actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device, beta, num_random)

    replay_buffer = ReplayBuffer(capacity=1000000)

    print(f"[Trial {trial_id}] Training Online Agent")
    ret_online = train_off_policy_agent(env_online, online_agent, num_episodes, replay_buffer, minimal_size, batch_size)
    print(f"[Trial {trial_id}] Training Offline SAC Agent")
    ret_offline1 = train_offline_agent(env_offline1, offline_agent1, replay_buffer, num_epochs, num_train_per_epoch, batch_size)
    print(f"[Trial {trial_id}] Training Offline CQL Agent")
    ret_offline2 = train_offline_agent(env_offline2, offline_agent2, replay_buffer, num_epochs, num_train_per_epoch, batch_size)

    return ret_online, ret_offline1, ret_offline2

def main():
    num_gpus = torch.cuda.device_count()
    trials_per_gpu = 1

    args = [(tid, (tid // trials_per_gpu) % num_gpus) for tid in range(num_trails)]
    with mp.Pool(processes=num_gpus * trials_per_gpu) as pool:
        results = pool.starmap(run_trial, args)

    online_returns, offline_sac_return, offline_cql_return = [], [], []
    for ret_online, ret_off1, ret_off2 in results:
        online_returns.append(ret_online)
        offline_sac_return.append(ret_off1)
        offline_cql_return.append(ret_off2)

    # Plot result
    plt.figure(figsize=(10, 6))
    plot_with_std(online_returns, 'Online SAC', 'blue')
    plot_with_std(offline_sac_return, 'Offline SAC', 'orange') 
    plot_with_std(offline_cql_return, 'Offline CQL-SAC', 'green')
    plt.title('Online SAC vs Offline Naive-SAC & CQL-SAC')
    plt.legend()
    plt.show()
    plt.savefig('Online SAC vs Offline Naive-SAC & CQL-SAC.png')

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()


