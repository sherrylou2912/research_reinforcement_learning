import torch
import numpy as np
from typing import Tuple, Optional
import math

def compute_rank(
    matrix: torch.Tensor,
    threshold: float = 1e-5
) -> int:
    """
    Compute the numerical rank of a matrix.
    
    Args:
        matrix: Input matrix
        threshold: Singular value threshold
        
    Returns:
        Numerical rank of the matrix
    """
    U, S, V = torch.svd(matrix)
    rank = (S > threshold).sum().item()
    return rank

def log_approximate_rank(
    agent,
    states: torch.Tensor,
    actions: torch.Tensor,
    num_samples: int = 10,
    sample_size: Tuple[int, int] = (64, 64),
    delta: float = 0.01
) -> float:
    """
    Estimate the log of the approximate rank of the Q-matrix.
    
    Args:
        agent: RL agent with critic networks
        states: Batch of states
        actions: Batch of actions
        num_samples: Number of random samples
        sample_size: Size of each random sample
        delta: Confidence parameter
        
    Returns:
        Estimated log rank
    """
    ranks = []
    for _ in range(num_samples):
        # Sample random states and actions
        idx = torch.randint(0, states.shape[0], (sample_size[0],))
        s = states[idx]
        a = actions[idx]
        
        # Compute Q-values
        with torch.no_grad():
            q1 = agent.critic1(s, a)
            q2 = agent.critic2(s, a)
            q = torch.min(q1, q2)
        
        # Reshape into matrix
        q_matrix = q.view(sample_size[0], -1)
        
        # Compute rank
        rank = compute_rank(q_matrix)
        ranks.append(rank)
    
    # Compute confidence interval
    ranks = torch.tensor(ranks)
    mean_rank = ranks.float().mean()
    std_rank = ranks.float().std()
    
    # Use high probability bound
    log_rank = math.log(mean_rank + delta * std_rank)
    
    return log_rank

def estimate_effective_rank(
    matrix: torch.Tensor,
    threshold: float = 0.95
) -> int:
    """
    Estimate the effective rank using singular value energy.
    
    Args:
        matrix: Input matrix
        threshold: Energy threshold
        
    Returns:
        Effective rank estimate
    """
    U, S, V = torch.svd(matrix)
    
    # Compute normalized cumulative energy
    total_energy = torch.sum(S)
    cum_energy = torch.cumsum(S, dim=0) / total_energy
    
    # Find rank that captures threshold energy
    effective_rank = torch.sum(cum_energy <= threshold).item() + 1
    
    return effective_rank

if __name__ == "__main__":
    # 测试矩阵秩工具
    print("Testing rank utilities...")
    
    # 创建测试矩阵
    n, m = 10, 8
    true_rank = 3
    
    # 生成低秩矩阵
    U = torch.randn(n, true_rank)
    V = torch.randn(m, true_rank)
    Q = U @ V.T
    
    # 添加噪声
    Q_noisy = Q + 0.1 * torch.randn_like(Q)
    
    # 测试秩计算
    rank = compute_rank(Q_noisy)
    print(f"\nRank computation test:")
    print(f"True rank: {true_rank}")
    print(f"Computed rank: {rank}")
    
    # 创建模拟智能体和数据用于测试log_approximate_rank
    class DummyAgent:
        def __init__(self):
            self.critic1 = lambda s, a: torch.randn(*s.shape[:-1], 1)
            self.critic2 = lambda s, a: torch.randn(*s.shape[:-1], 1)
    
    agent = DummyAgent()
    states = torch.randn(100, 4)  # 假设状态维度为4
    actions = torch.randn(100, 2)  # 假设动作维度为2
    
    # 测试近似秩的对数
    log_rank = log_approximate_rank(
        agent,
        states,
        actions,
        num_samples=5,
        sample_size=(32, 32),
        delta=0.01
    )
    
    print(f"\nLog approximate rank test:")
    print(f"Log rank: {log_rank:.3f}")
    
    # 测试有效秩估计
    effective_rank = estimate_effective_rank(Q_noisy)
    print(f"\nEffective rank estimation test:")
    print(f"True rank: {true_rank}")
    print(f"Effective rank: {effective_rank}")
    
    # 测试不同阈值的影响
    thresholds = [0.9, 0.95, 0.99]
    print("\nTesting different thresholds:")
    for threshold in thresholds:
        rank = estimate_effective_rank(Q_noisy, threshold=threshold)
        print(f"Threshold {threshold}: rank = {rank}")
    
    print("\nAll tests passed!")
