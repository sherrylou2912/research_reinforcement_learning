from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import pickle
#import d4rl

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self): 
        return len(self.buffer)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"✅ Replay buffer saved to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
        print(f"📂 Replay buffer loaded from {path}")

    def clear(self):
        self.buffer.clear()
        print("🗑️ Replay buffer cleared")
    
    @classmethod
    def from_d4rl(cls, env, capacity=1000000):
        dataset = env.get_dataset()
        N = dataset['rewards'].shape[0]

        capacity = N if capacity is None else min(capacity, N)
        buffer = cls(capacity)

        for i in range(N):
            state = dataset['observations'][i]
            action = dataset['actions'][i]
            reward = dataset['rewards'][i]
            next_state = dataset['next_observations'][i]
            done = bool(dataset['terminals'][i])
            buffer.add(state, action, reward, next_state, done)
        return buffer


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_offline_agent(env, agent,reply_buffer, num_epochs, num_train_per_epoch, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_epochs/10), desc = 'Iteration %d' % i) as pbar:
            for i_epoch in range(int(num_epochs/10)):
                epoch_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    state = next_state
                    epoch_return += reward 
                return_list.append(epoch_return)

                for _ in range(num_train_per_epoch):
                    b_s, b_a, b_r, b_ns, b_d = reply_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s, 
                        'actions': b_a, 
                        'next_states': b_ns, 
                        'rewards': b_r, 
                        'dones': b_d}
                    agent.update(transition_dict)
                
                if (i_epoch+1) % 10 == 0:
                    pbar.set_postfix({
                        'epoch': '%d' % (num_epochs/10 * i + i_epoch+1), 
                        'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list
def compute_q_matrix(agent, state_samples, action_samples):
    q_matrix = np.zeros((len(state_samples), len(action_samples)))
    with torch.no_grad():
        for i, s in enumerate(state_samples):
            s_tensor = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(agent.device)
            s_batch = s_tensor.repeat(len(action_samples), 1)  # [num_actions, state_dim]
            a_batch = torch.tensor(action_samples, dtype=torch.float).to(agent.device)  # [num_actions, action_dim]
            q_values = agent.critic1(s_batch, a_batch).squeeze().cpu().numpy()  # [num_actions]
            q_matrix[i] = q_values
    return q_matrix

def estimate_q_approx_rank(agent, replay_buffer, num_states=32, num_actions=32, threshold=0.01):
    # 随机采样 state 和 action
    b_s, _, _, _, _ = replay_buffer.sample(num_states)
    state_samples = b_s

    # 从动作空间均匀采样
    action_dim = agent.actor.fc_mu.out_features
    action_samples = np.random.uniform(low=-1.0, high=1.0, size=(num_actions, action_dim))

    q_matrix = compute_q_matrix(agent, state_samples, action_samples)
    rank = compute_approx_rank(q_matrix, threshold)
    return rank

def compute_approx_rank(q_matrix, threshold=0.01):
    u, s, vh = np.linalg.svd(q_matrix, full_matrices=False)
    total_energy = np.sum(s)
    cumulative_energy = np.cumsum(s)
    k = np.searchsorted(cumulative_energy, total_energy * (1 - threshold)) + 1
    return k
    
def log_q_matrix_rank(agent, replay_buffer, log_path, num_states=64, num_actions=64):
    rank = estimate_q_approx_rank(agent, replay_buffer, num_states, num_actions)
    with open(log_path, 'a') as f:
        f.write(f'{rank}\n')
    return rank

def train_offline_agent_with_rank(env, agent,reply_buffer, num_epochs, num_train_per_epoch, batch_size, log_path,sample_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_epochs/10), desc = 'Iteration %d' % i) as pbar:
            for i_epoch in range(int(num_epochs/10)):
                epoch_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    state = next_state
                    epoch_return += reward 
                return_list.append(epoch_return)

                for _ in range(num_train_per_epoch):
                    b_s, b_a, b_r, b_ns, b_d = reply_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s, 
                        'actions': b_a, 
                        'next_states': b_ns, 
                        'rewards': b_r, 
                        'dones': b_d}
                    agent.update(transition_dict)

                rank = log_q_matrix_rank(agent, reply_buffer, log_path)
                
                if (i_epoch+1) % 10 == 0:
                    pbar.set_postfix({
                        'epoch': '%d' % (num_epochs/10 * i + i_epoch+1), 
                        'return': '%.3f' % np.mean(return_list[-10:]),
                        'rank':'%.3f' % rank })
                pbar.update(1)
    return return_list

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)