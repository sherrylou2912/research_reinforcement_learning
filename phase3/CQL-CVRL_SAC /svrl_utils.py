import torch
import numpy as np
from torch.linalg import svd


def softimp(qmat_tensor, mask_prob=0.3, rank=None, n_iter=10, tol=1e-3, verbose=False):
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
    B, K = qmat_tensor.shape
    qmat = qmat_tensor.clone()

    # Mask random entries
    mask = (torch.rand(B, K, device=device) > mask_prob).float()
    qmat_masked = qmat * mask + torch.nan_to_num(qmat, nan=0.0) * (1 - mask)

    # Iterative SVD imputation
    Z = qmat_masked.clone()
    for i in range(n_iter):
        U, S, Vh = svd(Z, full_matrices=False)
        if rank is not None:
            S[rank:] = 0.0
        Z_new = (U @ torch.diag(S) @ Vh)
        Z_new = Z_new * mask + qmat_masked * (1 - mask)

        diff = torch.norm(Z_new - Z) / (torch.norm(Z) + 1e-6)
        if verbose:
            print(f"Iter {i+1}: diff = {diff:.6f}")
        Z = Z_new
        if diff < tol:
            if verbose:
                print("Converged.")
            break

    return Z


if __name__ == "__main__":
    torch.manual_seed(0)
    B, K, true_rank = 512, 64, 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create low-rank Q-matrix: Q = U @ V^T
    U = torch.randn(B, true_rank, device=device)
    V = torch.randn(K, true_rank, device=device)
    qmat = (U @ V.T).contiguous()  # shape [B, K], rank <= true_rank

    print("Original LOW-RANK matrix shape:", qmat.shape)
    qmat_recon = softimp_torch(qmat, mask_prob=0.3, rank=10, n_iter=30, verbose=True)
    print("Reconstructed matrix shape:", qmat_recon.shape)

    # Optional: check low-rank approximation quality
    error = torch.norm(qmat - qmat_recon) / torch.norm(qmat)
    print(f"Relative reconstruction error: {error.item():.4f}")
