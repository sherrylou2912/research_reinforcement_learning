# Low-Rank Structure in Deep Q-Learning / 深度Q学习中的低秩结构研究

## 1. Overview

| Setting | Description |
|----------|--------------|
| **Environment** | D4RL `halfcheetah-medium-v0``halfcheetah-random-v0 (in progress)``hopper-random (developing)``walker2d-random (developing)``hopper-medium (developing)`| `walker2d-medium (developing)`|
| **Algorithm base** | Soft Actor-Critic (offline variant) |
| **Metrics** | Test Reward, Test Success, Approx. Rank (Q-matrix) |
| **Hardware** | Apple M2 / CUDA GPU |
| **Training** | 200 epochs × 1000 updates/epoch, batch size 256 |

**Objective:**  
Investigate how *structured low-rank reconstruction* and *uncertainty-aware masking* improve offline SAC’s stability and extrapolation robustness.

## 2. Literature & Related Work

###  Structured Value-based RL (SVRL / SVP)
**Yang et al., ICLR 2020 — “Harnessing Structures for Value-based Planning and Reinforcement Learning.”**  


Key idea: reconstruct a partially observed batch-level Q sub-matrix using low-rank estimation (e.g., SoftImpute) before TD updates. Even with only ~20% observed entries, SVRL matches near-optimal planning results.  
→ Motivates the Q-matrix reconstruction component in SVRL-SAC.

---

###  Uncertainty-Aware Low-Rank Q-Matrix Estimation (UA-LQE)
**Sang et al., DAI 2021 — “Uncertainty-Aware Low-Rank Q-Matrix Estimation for Deep Reinforcement Learning.”**  
Proposes combining uncertainty estimation (via bootstrapped ensembles or count-based variance) with low-rank matrix reconstruction

The method constructs a Q-matrix \( Q_B \) per batch, estimates elementwise uncertainty \( U_B = \text{Std}_m(Q_m) \), and masks the top-p% most uncertain entries before applying low-rank completion:  
\[
Q_B^{rec} = \text{SoftImpute}_\zeta(Q_B \odot M)
\]

This links uncertainty concentration to Q-matrix rank, showing that large singular values often correspond to OOD-induced overestimation.  

→ Directly informs the **uncertainty-masked reconstruction** mechanism used in UALQE-SAC.

---

###  Uncertainty-Based Offline RL with Diversified Q-Ensemble (EDAC)
**An et al., NeurIPS 2021 — “Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble.”**  

Introduces a diversity-regularized ensemble for offline RL, mitigating overestimation in OOD regions.  
By adding a gradient decorrelation penalty between Q-functions:  
\[
L_{EDAC} = L_{TD} + \beta \sum_{m \ne n} \|\nabla_a Q_m - \nabla_a Q_n\|^2
\]

EDAC achieves uncertainty robustness comparable to large bootstrap ensembles but with fewer networks.  
→ Inspires the EDUALQE variant, replacing bootstrap with EDAC-style ensemble regularization for stable uncertainty in offline settings.

---

###  Conceptual Connection to Current Work
- **SVRL** provides the foundation for **structured Q-matrix reconstruction** and rank diagnostics.  
- **UA-LQE** introduces the concept of using **uncertainty as a mask signal** for selective reconstruction.  
- **EDAC** offers a principled way to obtain **well-calibrated uncertainty** through gradient diversification.  
Together, these methods motivate a unified framework — **EDUALQE-SAC** — combining **low-rank regularization** with **stable uncertainty masking** for offline RL.


### Potential Reference Papers 

#### A. Robust PCA / Low-Rank + Sparse Decomposition (for replacing SoftImpute)
- **Candès, Li, Ma, Wright (2011)** — *Robust Principal Component Analysis?* (JACM).  
  Canonical Principal Component Pursuit (PCP): decomposes \(X=L+S\) with nuclear-norm + \(\ell_1\) to separate low-rank signal from sparse outliers. Strong theoretical guarantees; ideal drop-in to handle OOD-induced spikes in \(Q\)-matrices.
- **Chandrasekaran, Sanghavi, Parrilo, Willsky (2011)** — *Rank-Sparsity Incoherence for Matrix Decomposition* (SIAM Review).  

- **Xu, Caramanis, Mannor (2012)** — *Outlier Pursuit* (IEEE TIT).  

- **Lin, Chen, Ma (2010)** — *Augmented Lagrange Multiplier (ALM) Method for RPCA*.  

- **Zhou, Tao (2011)** — *GoDec: Randomized Low-rank & Sparse Matrix Decomposition* (ICML).  

- **Mazumder, Hastie, Tibshirani (2010)** — *Spectral Regularization for Large Incomplete Matrices (Soft-Impute)* 

---



## 3. Algorithmic Timeline

### (1) Naive SAC (Offline Baseline)
**Goal:** Reproduce baseline offline SAC performance.  
**Setup:**
- Actor–Critic: 2×256 MLP, τ = 0.005, fixed replay buffer.  
**Result:** Reward ≈ 2500; unstable Q-rank (2.0–4.0); clear overestimation from OOD actions.

---

### (2) CQL-SAC (Conservative Q-Learning)
**Goal:** Add conservative penalty to discourage OOD action overestimation.  
**Loss:**  
\[
L_{CQL} = \alpha (\log \sum_a e^{Q(s,a)} - Q(s,a_{data}))
\]
with adaptive α (Lagrange update).  
**Result:** Reward ≈ 6000; smooth convergence; Avg Rank ≈ 12–15.  
→ Serves as strong offline baseline.

---

### (3) SVRL-SAC (Structured Value-based RL)
**Goal:** Leverage low-rank structure in \( Q(s,a) \) for extrapolation correction.  
**Q-matrix construction:**
- \( Q_B ∈ ℝ^{B×B} \): rows = sampled states \(s_i\), columns = actions \(a_j\) from current policy.  
- Diagonal entries \(Q(s_i,a_i)\) observed; 40 % random entries masked.  

**Reconstruction (SoftImpute):**
\[
Q_B^{(t+1)} = \text{Shrink}_\zeta\big(P_\Omega(Q_B) + P_{\Omega^c}(U_t V_t^\top)\big)
\]
with **ζ = 50** shrinkage.  
Updated critic using reconstructed diagonals \(Q_B^{rec}(s_i,a_i)\).

**Result:**
- Reward ≈ 6000 (close to CQL).  
- Rank ≈ 23–25 → highest among all.  
- High variance across seeds.  
✅ Demonstrates structured recovery and extrapolation control even offline.

---

### (4) UALQE-SAC (Bootstrap Ensemble)
**Goal:** Combine uncertainty estimation with matrix reconstruction.  
**Setup:**
- 5-critic ensemble, bootstrap sampling per batch.  
- Uncertainty:
  \[
  U(s_i,a_j) = \text{Std}_m(Q_m(s_i,a_j)), \quad m=1..5
  \]
- Top 20 % high-uncertainty entries masked pre-SoftImpute (ζ = 50).  

**Observation:**  
- Early learning surge, collapse after ≈ 150 epochs.  
- Rank slightly lower than SVRL, highly unstable.  
❌ Bootstrap variance amplifies noise in static offline data.

---

### (5) EDUALQE (EDAC-Style Ensemble)
**Goal:** Replace bootstrap with *EDAC-style diversity regularization* to stabilize uncertainty.  
**Implementation:**
- Shared target network across ensemble members.  
- Loss:
  \[
  L = L_{TD} + β \sum_{m≠n}\|\nabla_a Q_m(s,a) - \nabla_a Q_n(s,a)\|^2
  \]
  with β = 0.01.  
- Same SoftImpute + uncertainty mask pipeline.

**Result:**  
- Reward ≈ 3000, smooth curve.  
- Rank ≈ 16–20.  
✅ Uncertainty stabilized, no collapse observed.


---

## 4. Key Takeaways

1. **Matrix reconstruction (SVRL)** reliably enforces structure and mitigates extrapolation.  
2. **Bootstrap-based uncertainty (UALQE)** is unstable offline — uncertainty too noisy.  
3. **EDAC regularization + EMA smoothing** provide effective uncertainty control.  
4. **SoftImpute shrinkage ζ≈50** is robust; adaptive decay helps fine-tuning.  
5. **Rank trend** acts as an interpretability and stability diagnostic (< 10 = healthy).  

---

## 5. Next Steps

- **Cross-environment validation:**  
  Extend **EDUALQE** experiments to additional D4RL benchmarks, including `antmaze-medium` and `hopper-medium`, to verify generality and robustness across task types and reward sparsity levels.

- **Singular value and uncertainty analysis:**  
  Investigate the **dominant first singular value (σ₁)** observed during training.  
  This unusually high σ₁ may reflect *global bias components* or *OOD-driven over-optimism*.  
  Conduct a joint visualization of **uncertainty heatmaps** and **σ₁–u₁–v₁ trajectories** across training epochs and tasks to better understand the relationship between **spectral structure** and **uncertainty concentration in OOD regions**.

- **Robust reconstruction module:**  
  Replace the standard **SoftImpute** operator with **Robust PCA (Principal Component Pursuit)** to improve resilience against heavy-tailed noise and biased Q estimates, ensuring more stable low-rank recovery under corrupted or highly uncertain entries.

---

*Last updated: Oct 6 2025*

---