import torch
import numpy as np
from typing import Optional, Tuple

def softimp(
    Q: torch.Tensor,
    mask_prob: float = 0.5,
    rank: int = 10,
    n_iter: int = 30,
    threshold: float = 1e-5
) -> torch.Tensor:
    """
    Soft impute algorithm for matrix completion with automatic masking.
    
    Args:
        Q: Input Q-value matrix [batch_size, n_actions]
        mask_prob: Probability of masking entries
        rank: Target rank for the completed matrix
        n_iter: Maximum number of iterations
        threshold: Convergence threshold
        
    Returns:
        Completed low-rank matrix
    """
    device = Q.device
    mask = torch.rand_like(Q) > mask_prob
    
    # Initialize with mean of observed entries
    Q_obs = Q * mask
    mean_val = Q_obs.sum() / mask.sum()
    X_old = Q_obs.clone()
    X_old[~mask] = mean_val
    
    for _ in range(n_iter):
        # SVD and soft thresholding
        U, S, V = torch.svd(X_old)
        S_thresh = torch.clamp(S[:rank], min=0)
        X_new = U[:, :rank] @ torch.diag(S_thresh) @ V[:, :rank].T
        
        # Project onto observed entries
        X_new[mask] = Q_obs[mask]
        
        # Check convergence
        diff = torch.norm(X_new - X_old) / torch.norm(X_old)
        if diff < threshold:
            break
            
        X_old = X_new
    
    return X_new

def softimp_ua(
    Q: torch.Tensor,
    mask: torch.Tensor,
    rank: int = 10,
    n_iter: int = 30,
    threshold: float = 1e-5
) -> torch.Tensor:
    """
    Uncertainty-aware soft impute algorithm for matrix completion.
    
    Args:
        Q: Input Q-value matrix [batch_size, n_actions]
        mask: Binary mask tensor indicating valid entries
        rank: Target rank for the completed matrix
        n_iter: Maximum number of iterations
        threshold: Convergence threshold
        
    Returns:
        Completed low-rank matrix
    """
    device = Q.device
    
    # Initialize with mean of observed entries
    Q_obs = Q * mask
    mean_val = Q_obs.sum() / mask.sum()
    X_old = Q_obs.clone()
    X_old[~mask] = mean_val
    
    for _ in range(n_iter):
        # SVD and soft thresholding
        U, S, V = torch.svd(X_old)
        S_thresh = torch.clamp(S[:rank], min=0)
        X_new = U[:, :rank] @ torch.diag(S_thresh) @ V[:, :rank].T
        
        # Project onto observed entries with uncertainty weighting
        X_new = mask * Q + (1 - mask) * X_new
        
        # Check convergence
        diff = torch.norm(X_new - X_old) / torch.norm(X_old)
        if diff < threshold:
            break
            
        X_old = X_new
    
    return X_new

if __name__ == "__main__":
    # 测试基本的软填充
    print("Testing basic soft impute...")
    
    # 创建测试矩阵
    n, m = 10, 8
    true_rank = 3
    
    # 生成低秩矩阵
    U = torch.randn(n, true_rank)
    V = torch.randn(m, true_rank)
    Q = U @ V.T
    
    # 添加噪声
    Q_noisy = Q + 0.1 * torch.randn_like(Q)
    
    # 测试基本的软填充
    Q_completed = softimp(Q_noisy, mask_prob=0.3, rank=true_rank)
    error = torch.norm(Q - Q_completed).item()
    print(f"Basic soft impute reconstruction error: {error}")
    
    # 测试不确定性感知的软填充
    print("\nTesting uncertainty-aware soft impute...")
    
    # 创建不确定性掩码
    mask = torch.rand_like(Q) > 0.3
    
    # 应用不确定性感知的软填充
    Q_completed_ua = softimp_ua(Q_noisy, mask, rank=true_rank)
    error_ua = torch.norm(Q - Q_completed_ua).item()
    print(f"Uncertainty-aware soft impute reconstruction error: {error_ua}")
    
    # 比较两种方法
    print("\nComparison:")
    print(f"Original matrix shape: {Q.shape}")
    print(f"Original matrix rank: {true_rank}")
    print(f"Basic completion error: {error}")
    print(f"UA completion error: {error_ua}")
    
    # 验证重建矩阵的秩
    _, S1, _ = torch.svd(Q_completed)
    _, S2, _ = torch.svd(Q_completed_ua)
    effective_rank1 = torch.sum(S1 > 1e-5).item()
    effective_rank2 = torch.sum(S2 > 1e-5).item()
    
    print(f"\nEffective ranks:")
    print(f"Basic completion: {effective_rank1}")
    print(f"UA completion: {effective_rank2}")
