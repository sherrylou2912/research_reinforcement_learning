import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent import SAC, CQL_SAC
from rl_utils import ReplayBuffer, train_offline_agent_with_rank
import os 
import gym as gym
import d4rl

# Experiment parameters
env_name = "hopper-medium-v1"
num_trails = 3
num_epoch = 100
num_train_per_epoch = 100
batch_size = 128
sample_size = 64*64

# Hyperparameters
actor_lr = 1e-4
critic_lr = 1e-3
alpha_lr = 3e-4
hidden_dim = 256
gamma = 0.99
tau = 0.005
beta = 3.0
num_random = 10
capacity = int(1e6)


def make_env():
    print("✅ Using new_step_api=True")
    print("✅ env_name passed in is:", env_name)
    env = gym.make(env_name, new_step_api=True, disable_env_checker=True)
    return env


def plot_with_std(data, label, color):
    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    x = np.arange(len(mean))
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

def run_trial(trial_id, log_dir):
    env = make_env()
    dataset_env = make_env()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bounds = env.action_space.high[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_entropy = - action_dim

    # Create the agents
    sac_agent = SAC(state_dim, hidden_dim, action_dim, action_bounds,
                    actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)

    cql_agent = CQL_SAC(state_dim, hidden_dim, action_dim, action_bounds,
                        actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device, beta, num_random)

    replay_buffer = ReplayBuffer.from_d4rl(dataset_env, capacity=capacity)

    log_sac = os.path.join(log_dir, f"rank_sac_trial_{trial_id}.txt")
    log_cql = os.path.join(log_dir, f"rank_cql_trial_{trial_id}.txt")

    print(f"[Trial {trial_id}]")
    print('Training SAC agent')
    ret_sac = train_offline_agent_with_rank(env, sac_agent, replay_buffer, num_epoch, num_train_per_epoch, batch_size, log_sac, sample_size)
    print('Training CQL agent')
    ret_cql = train_offline_agent_with_rank(env, cql_agent, replay_buffer, num_epoch, num_train_per_epoch, batch_size, log_cql, sample_size)

    return ret_sac, ret_cql

def main():
    log_dir = "hopper_logs"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("results_hopper", exist_ok=True)

    all_ret_sac = []
    all_ret_cql = []

    for t in range(num_trails):
        ret_sac, ret_cql = run_trial(t, log_dir)
        all_ret_sac.append(ret_sac)
        all_ret_cql.append(ret_cql)

        np.savez(f"results_hopper/trail_{t}_return.npz", sac=ret_sac, cql=ret_cql)

    np.savez("results_hopper/returns.npz", sac=all_ret_sac, cql=all_ret_cql)

    plt.figure(figsize=(10, 6))
    plot_with_std(all_ret_sac, "SAC", "blue")
    plot_with_std(all_ret_cql, "CQL-SAC", "red")
    plt.title("SAC vs CQL-SAC on hopper-medium-v2")
    plt.xlabel("Epoch")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig("results_hopper/returns.png")
    plt.show()

if __name__ == "__main__":
    main()




