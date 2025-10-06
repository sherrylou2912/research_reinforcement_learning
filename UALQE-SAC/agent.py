import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor, CriticEnsemble
from svrl_utils import softimp


class UALQESAC(nn.Module):
    """Uncertainty-Aware Low-rank Q-matrix Estimation SAC (Twin-Q + N×N shared columns)."""

    def __init__(
        self,
        # SAC
        state_size,
        action_size,
        tau,
        hidden_size,
        learning_rate,
        # UALQE
        n_q_networks,        # ensemble size for uncertainty (M >= 2; M >= 5 recommended)
        eta,                 # kept for compatibility
        zeta,                # soft-impute shrinkage (paper ~ 50)
        n_action_sample,     # kept for compatibility (unused in N×N path)
        lambda_structure,    # set 1.0 when using reconstructed diagonal only
        # software
        device,
    ):
        super().__init__()

        # ---- SAC hyper/housekeeping ----
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = torch.tensor(0.99, dtype=torch.float32, device=device)
        self.tau = tau
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.clip_grad_param = 5.0  # slightly looser for stability

        self.target_entropy = torch.tensor(-action_size, dtype=torch.float32, device=device)  # -dim(A)
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)  # consider smaller LR (e.g., 3e-4)

        # ---- UALQE params ----
        self.eta = eta
        self.n_q_networks = n_q_networks         # M
        self.lambda_struct = lambda_structure     # expect 1.0 in this variant
        self.zeta = zeta
        self.n_action_sample = n_action_sample    # unused here (kept for BC)
        self._unc_ema = None 
        self.unc_ema_beta = 0.99
        self._q_row_prev = None

        #bootstrap 
        self.poisson_lambda = 0.7  
        self.bernoulli_p = 0.7
        

        # --- New Knobs ---
        self.global_steps = 0
        self.ens_delay = 5
        self.tau_ens = 5e-4

            #mask schedule 
        self.keep_prob_target1 = 0.9 
        self.keep_prob_target2 = 0.8
        self.keep_warmup_steps = 5000
        self.keep_ramp_steps = 5000
        self.keep_late_step = 10000 


        # ---- Actor ----
        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # ---- Uncertainty ensemble (M critics) ----
        self.uncertainty_ensemble = CriticEnsemble(
            state_size, action_size, hidden_size, self.n_q_networks
        ).to(self.device)
        self.uncertainty_ensemble_target = CriticEnsemble(
            state_size, action_size, hidden_size, self.n_q_networks
        ).to(self.device)

        # ---- Sampling Twin-Q for reconstruction path ----
        self.sampling_critic1 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.sampling_critic2 = Critic(state_size, action_size, hidden_size).to(self.device)
        self.sampling_critic1_target = Critic(state_size, action_size, hidden_size).to(self.device)
        self.sampling_critic2_target = Critic(state_size, action_size, hidden_size).to(self.device)

        # ---- Init targets ----
        self.uncertainty_ensemble_target.load_state_dict(self.uncertainty_ensemble.state_dict())
        self.sampling_critic1_target.load_state_dict(self.sampling_critic1.state_dict())
        self.sampling_critic2_target.load_state_dict(self.sampling_critic2.state_dict())

        # ---- Optimizers ----
        self.uncertainty_ensemble_optimizer = optim.Adam(self.uncertainty_ensemble.parameters(), lr=self.learning_rate)
        self.sampling_critic1_optimizer = optim.Adam(self.sampling_critic1.parameters(), lr=self.learning_rate)
        self.sampling_critic2_optimizer = optim.Adam(self.sampling_critic2.parameters(), lr=self.learning_rate)

    # -------------------- API --------------------

    def get_action(self, state, eval=False):
        """Returns action for given state under current policy. Expects numpy state."""
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action = self.actor_local.get_det_action(state) if eval else self.actor_local.get_action(state)
        return action.cpu().numpy()

    # -------------------- Helpers --------------------

    def calc_policy_loss(self, states):
        """
        Standard SAC policy loss with Twin-Q min:
          J_pi = E_s [ alpha * log_pi(a|s) - min(Q1(s,a), Q2(s,a)) ]
        Returns: (actor_loss, log_pis)
        """
        actions_pred, log_pis = self.actor_local.evaluate(states)        # (B,A), (B,1)
        q1 = self.sampling_critic1(states, actions_pred)                 # (B,1)
        q2 = self.sampling_critic2(states, actions_pred)                 # (B,1)
        q_min = torch.min(q1, q2)
        alpha_detached = self.log_alpha.exp().detach()
        actor_loss = (alpha_detached * log_pis - q_min).mean()
        return actor_loss, log_pis

    def _compute_policy_values(self, obs_pi, obs_q, with_ensemble=True):
        """
        Diagnostics helper. Returns dict with actions, log_pis, q1, q2, q_min, v_pi, and optional ensemble mean/std.
        """
        with torch.no_grad():
            actions_pred, log_pis = self.actor_local.evaluate(obs_pi)
            q1 = self.sampling_critic1(obs_q, actions_pred)
            q2 = self.sampling_critic2(obs_q, actions_pred)
            q_min = torch.min(q1, q2)
            v_pi = q_min - self.log_alpha.exp() * log_pis

            ens_mean = ens_std = None
            if with_ensemble:
                ens = self.uncertainty_ensemble(obs_q, actions_pred).squeeze(-1)  # (M,B)
                ens_mean = ens.mean(dim=0, keepdim=True).transpose(0, 1)          # -> (B,1)
                ens_std  = ens.std(dim=0,  keepdim=True).transpose(0, 1)          # -> (B,1)

            return {
                "actions": actions_pred, "log_pis": log_pis,
                "q1": q1, "q2": q2, "q_min": q_min, "v_pi": v_pi,
                "ens_mean": ens_mean, "ens_std": ens_std,
            }

    def _compute_random_values(self, obs, actions):
        """Optional CQL-style diagnostic: returns Q1, Q2, and min without entropy."""
        with torch.no_grad():
            q1 = self.sampling_critic1(obs, actions)
            q2 = self.sampling_critic2(obs, actions)
            q_min = torch.min(q1, q2)
            return q1, q2, q_min

    # -------------------- Core Learn --------------------

    def learn(self, experiences):
        """
        Twin-Q + N×N shared columns + per-row top-20% uncertainty masking + Soft-Impute(zeta=self.zeta, n_iter=100).
        Uses reconstructed diagonal ONLY as TD target (lambda_structure expected to be 1.0).
        experiences: (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = experiences
        B = next_states.size(0)
        assert B >= 2, "Batch size too small for N×N shared-columns construction."
        M = self.n_q_networks
        assert M >= 2, "Uncertainty ensemble size must be >= 2 to compute std-based uncertainty."

        # ---------------- Actor update (now via helper) ----------------
        actor_loss, log_pis = self.calc_policy_loss(states)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor_local.parameters(), self.clip_grad_param)
        self.actor_optimizer.step()

        # Temperature (alpha) update
        alpha = self.log_alpha.exp()
        alpha_loss = -(alpha * (log_pis.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        current_alpha = self.log_alpha.exp().detach()

        # ---------------- Build N×N shared-columns grid & reconstruct diagonal ----------------
        with torch.no_grad():
            # Shared columns: actions for each next_state under current policy
            a_cols, next_log_pis = self.actor_local.evaluate(next_states)      # (B,A), (B,1)

            # Build (B×B) grid: rows are s'_i, columns are shared actions a'_j
            S_grid = next_states.unsqueeze(1).expand(B, B, next_states.size(-1)).reshape(B * B, -1)   # (B*B, S)
            A_grid = a_cols.unsqueeze(0).expand(B, B, a_cols.size(-1)).reshape(B * B, -1)             # (B*B, A)

            # Twin target critics on the grid, then take min -> (B,B)
            q1_grid = self.sampling_critic1_target(S_grid, A_grid).squeeze(-1)  # (B*B,)
            q2_grid = self.sampling_critic2_target(S_grid, A_grid).squeeze(-1)  # (B*B,)
            qmin_grid  = torch.min(q1_grid, q2_grid).view(B, B)                     # (B,B)

            # Uncertainty on the SAME (B,B) grid using ensemble target: std over critics
            q_ens = self.uncertainty_ensemble_target(S_grid, A_grid).squeeze(-1)  # (M, B*B)
            assert q_ens.dim() == 2 and q_ens.size(0) == M and q_ens.size(1) == B * B, \
                f"uncertainty_ensemble_target must output (M, B*B), got {tuple(q_ens.shape)}"
            unc = q_ens.view(M, B, B).std(dim=0)                                   # (B,B)

            # Per-row top-20% most-uncertain -> set to missing for imputation (mask=1 keep, 0 missing)

            # --- Mask schedule ---
            if self.global_steps < self.keep_warmup_steps:
                p_keep = 1.0
            elif self.global_steps < self.keep_warmup_steps + self.keep_ramp_steps:
                t = (self.global_steps - self.keep_warmup_steps) / self.keep_ramp_steps
                p_keep = 1.0 - t * (1.0 - self.keep_prob_target1)
            elif self.global_steps < self.keep_late_step:
                p_keep = self.keep_prob_target1
            else:
                p_keep = self.keep_prob_target2

            # --- EMA on uncertainty for stability ---
            if (self._unc_ema is None) or (self._unc_ema.shape != unc.shape):
                self._unc_ema = unc
            else:
                self._unc_ema = self.unc_ema_beta * self._unc_ema + (1 - self.unc_ema_beta) * unc
            unc_thr = self._unc_ema

            # --- Row-wise quantile thresholding ---
            q_row_now = torch.quantile(unc_thr, p_keep, dim=1, keepdim=True)   # (B,1)
            if (self._q_row_prev is None) or (self._q_row_prev.shape != q_row_now.shape):
                q_row = q_row_now
            else:
                q_row = 0.7 * self._q_row_prev + 0.3 * q_row_now
            self._q_row_prev = q_row

            mask = (unc_thr <= q_row).float()                      # (B,B)                                            # (B,B)

            # Optional: standardize Q_grid per batch for SVD stability (commented)
            #q_mean = Q_grid.mean()
            #q_std  = Q_grid.std().clamp_min(1e-6)
            #Q_grid_norm = (Q_grid - q_mean) / q_std
            #Q_recon_norm = softimp(Q_grid_norm, mask=mask, zeta=self.zeta, n_iter=100)
            #Q_recon = Q_recon_norm * q_std + q_mean

            Q_target_recon = softimp(qmin_grid, mask=mask, zeta=self.zeta, n_iter=100)          # (B,B)
            lo = torch.quantile(Q_target_recon, 0.01, dim = 1, keepdim = True)
            hi = torch.quantile(Q_target_recon, 0.99, dim = 1, keepdim = True)
            Q_target_recon = torch.clamp(Q_target_recon, lo, hi)
            q_diag = Q_target_recon.diag()      
            
            rewards = rewards.view(-1) 
            dones   = dones.view(-1)                                                 # (B,)

            # TD target using reconstructed diagonal only (lambda_structure == 1.0)
            Q_targets_sampling = rewards + self.gamma * (1 - dones)* (q_diag)  # (B,)
            target_s = Q_targets_sampling.detach() 
            assert target_s.shape == (B,), f"target_s shape mismatch: got {target_s.shape}, expected {(B,)}"

        # ---------------- Sampling Twin-Q losses ----------------
        q1_sa = self.sampling_critic1(states, actions).squeeze(-1)     # (B,)
        q2_sa = self.sampling_critic2(states, actions).squeeze(-1)  # (B,)
        assert q1_sa.shape == q2_sa.shape == target_s.shape == (B,), f"{q1_sa.shape=}, {q2_sa.shape=}, {target_s.shape=}"

        sampling_c1_loss = F.mse_loss(q1_sa, target_s)
        sampling_c2_loss = F.mse_loss(q2_sa, target_s)

        self.sampling_critic1_optimizer.zero_grad()
        sampling_c1_loss.backward()
        clip_grad_norm_(self.sampling_critic1.parameters(), self.clip_grad_param)
        self.sampling_critic1_optimizer.step()

        self.sampling_critic2_optimizer.zero_grad()
        sampling_c2_loss.backward()
        clip_grad_norm_(self.sampling_critic2.parameters(), self.clip_grad_param)
        self.sampling_critic2_optimizer.step()

        # ---------------- Uncertainty ensemble loss (vectorized bootstrap) ----------------
        do_update_ens = (self.global_steps % self.ens_delay == 0)

        with torch.no_grad():
            # target members在对角动作上的Q（无熵）
            ens_target_q = self.uncertainty_ensemble_target(next_states, a_cols).squeeze(-1)  # (M,B)
            ens_targets  = rewards.unsqueeze(0) + self.gamma * (1 - dones).unsqueeze(0) * ens_target_q  # (M,B)
            assert ens_targets.shape == (self.n_q_networks, states.size(0))

        # 统一到 (M,B) 形状，避免后面广播/索引复杂度
        ens_pred = self.uncertainty_ensemble(states, actions).squeeze(-1)   # (M,B)

        # ----- Poisson(1) 向量化 bootstrap 权重 -----
        #W = torch.poisson(torch.full_like(ens_pred, self.poisson_lambda))   # (M,B), {0,1,2,...}, float
        #row_sums = W.sum(dim=1).clamp_min(1.0)         # (M,), 防除零，简单稳妥

        # ----- Bernoulli(p) 向量化 bootstrap 权重 -----
        W = (torch.rand_like(ens_pred) < self.bernoulli_p).float()  # (M,B), {0,1}, float
        row_sums = W.sum(dim=1).clamp_min(1.0)

        # 加权MSE（每个成员自己的子样本）
        diff  = ens_pred - ens_targets.detach()        # (M,B)
        lossM = ((diff ** 2) * W).sum(dim=1) / row_sums
        uncertainty_loss = lossM.mean()

        if do_update_ens:
            self.uncertainty_ensemble_optimizer.zero_grad()
            uncertainty_loss.backward()
            clip_grad_norm_(self.uncertainty_ensemble.parameters(), self.clip_grad_param)
            self.uncertainty_ensemble_optimizer.step()



        # ---------------- Soft update targets ----------------
        self.soft_update(self.sampling_critic1, self.sampling_critic1_target,self.tau)
        self.soft_update(self.sampling_critic2, self.sampling_critic2_target,self.tau)
        if do_update_ens:
            self.soft_update(self.uncertainty_ensemble, self.uncertainty_ensemble_target,self.tau_ens)

        avg_sampling_loss = 0.5 * (sampling_c1_loss + sampling_c2_loss)

        self.global_steps += 1
        return (float(actor_loss.item()),
                float(alpha_loss.item()),
                float(avg_sampling_loss.item()),
                float(uncertainty_loss.item()),
                float(qmin_grid.std().detach()),
                float(qmin_grid.abs().max().detach()),
                float(Q_target_recon.std().item()),
                float(Q_target_recon.abs().max().item()),
                float(current_alpha))

    # -------------------- Utils --------------------

    def soft_update(self, local_model, target_model,tau):
        """θ_target ← τ·θ_local + (1-τ)·θ_target"""
        for t_param, l_param in zip(target_model.parameters(), local_model.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)



