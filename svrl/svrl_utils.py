import torch
import numpy as np
from torch.linalg import svd


def softimp(qmat_tensor,  mask = None, mask_prob=0.3, n_iter=10, zeta = 30.0, 
            tol=1e-3, verbose=False):
    """
    Efficient SoftImpute using PyTorch (GPU-compatible, no normalization).
    Args:
        qmat_tensor: [B, K] float tensor (on CPU or GPU)
        mask_prob: float, percent of entries to mask (simulate uncertainty)
        rank: int, number of singular values to retain (if None, auto rank)
        n_iter: int, max number of iterations
        tol: float, convergence threshold
        verbose: bool, print convergence info

    Returns:
        reconstructed_qmat: [B, K] tensor with missing entries imputed
    """
    device = qmat_tensor.device
    M = qmat_tensor.clone()

    if mask is None:
        mask = (torch.rand_like(M) > mask_prob).float()

    M_old = M * mask 

    # Iterative SVD imputation
    for i in range(n_iter):
        matrix_for_svd = mask * M_old + (1-mask) * M

        U,S,V = torch.svd(matrix_for_svd, some = False)

        if len(S) > 0 and S[0] > 0:
            lambda_val = S[0]/zeta
        else: 
            lambda_val = 0.0

        S_lambda = torch.clamp(S - lambda_val, min = 0.0)
        
        diag_size = min(U.shape[1], V.shape[1])
        S_diag = torch.zeros(U.shape[1], V.shape[1], device = device)

        for  j in range(min(diag_size, len(S_lambda))):
            S_diag[j, j] = S_lambda[j]

        M_new = U @ S_diag @ V.t()

        relative_change = torch.norm(M_new - M_old, 'fro') ** 2 / (torch.norm(M_old, 'fro') ** 2 + 1e-8)
        
        if verbose and i % 10 == 0:
            print(f"Iteration {i+1}, relative change: {relative_change.item():.6f}")
            print(f"Original S length: {len(S)}, Non-zero S_lambda: {(S_lambda > 1e-6).sum().item()}")

        if relative_change < tol:
            if verbose:
                print(f" Converged after {i+1} iterations")
            return M_new

        M_old = M_new.clone()

    return M_old 


if __name__ == "__main__":
    torch.manual_seed(0)
    B, K, true_rank = 512, 64, 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create low-rank Q-matrix: Q = U @ V^T
    U = torch.randn(B, true_rank, device=device)
    V = torch.randn(K, true_rank, device=device)
    qmat = (U @ V.T).contiguous()  # shape [B, K], rank <= true_rank
    noise = 0.1 * torch.randn_like(qmat)
    qmat_noisy = qmat + noise 
    print("Original LOW-RANK matrix shape:", qmat_noisy.shape)
    qmat_recon = softimp(qmat, mask_prob=0.3, rank=10, n_iter=30, verbose=True)
    print("Reconstructed matrix shape:", qmat_recon.shape)

    # Optional: check low-rank approximation quality
    error = torch.norm(qmat - qmat_recon) / torch.norm(qmat)
    print(f"Relative reconstruction error: {error.item():.4f}")
