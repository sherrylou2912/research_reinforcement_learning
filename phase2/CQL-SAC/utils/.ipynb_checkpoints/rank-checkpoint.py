import numpy as np
from numpy.linalg import svd
import torch

def approximate_rank(Q, delta = 0.01):
    U,S,V = svd(Q, full_matrices = False)
    total_sum = np.sum(S)
    cumulative_sum = np.cumsum(S)
    threshold = (1 - delta) * total_sum

    arank = np.searchsorted(cumulative_sum, threshold) + 1
    return arank

def sample_q_submatrix(agent, states, actions, sample_size = (128,128)):
    device = states.device

    states = states.to(device)
    actions = actions.to(device)
    
    num_states = min(sample_size[0], states.shape[0])
    num_actions = min(sample_size[1], actions.shape[0])

    state_indices = torch.randint(0, states.shape[0], (num_states,), device=device)
    action_indices = torch.randint(0, actions.shape[0], (num_actions,), device=device)

    sampled_states = states[state_indices]  # (num_states, state_dim)
    sampled_actions = actions[action_indices]  # (num_actions, action_dim)

    # Compute Q(s, a) for all (s, a) pairs → Cartesian product
    state_expand = sampled_states.unsqueeze(1).repeat(1, num_actions, 1).view(-1, states.shape[1])
    action_expand = sampled_actions.unsqueeze(0).repeat(num_states, 1, 1).view(-1, actions.shape[1])

    with torch.no_grad():
        q_values = agent.critic1(state_expand, action_expand).view(num_states, num_actions)

    return q_values.cpu().numpy()

def log_approximate_rank(agent, states, actions, wandb_step, num_samples=10, sample_size=(64, 64), delta=0.01):
    """
    Compute and log the empirical average approximate rank of Q-matrix using replay buffer samples.

    Params:
        agent: SAC/CQL agent
        replay_buffer: experience buffer (must have states & actions)
        wandb_step: wandb logging step
        num_samples: how many submatrices to sample for averaging
        sample_size: submatrix size (|S|, |A|)
        delta: threshold for approximate rank
    """
    # Ensure states & actions are torch.Tensor on correct device
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states).float().to(agent.device)
    else:
        states = states.to(agent.device)

    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions).float().to(agent.device)
    else:
        actions = actions.to(agent.device)
    ranks = []
    for _ in range(num_samples):
        Q_submatrix = sample_q_submatrix(agent, states, actions, sample_size)
        arank = approximate_rank(Q_submatrix, delta=delta)
        ranks.append(arank)

    avg_rank = np.mean(ranks)

    return avg_rank
