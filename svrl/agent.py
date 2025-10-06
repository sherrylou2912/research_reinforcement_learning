import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor
from svrl_utils import softimp


class SVRLSAC(nn.Module):
    """SAC + SVRL-style low-rank target with N×N shared-column Q-matrix."""

    def __init__(self,
                 state_size,
                 action_size,
                 tau,
                 hidden_size,
                 learning_rate,
                 mask_prob,       # kept for back-compat (now using _mask_prob_at)
                 lambda_struct,   # kept for back-compat (now scheduled)
                 zeta,            # kept for back-compat (now adaptive & capped)
                 device):
        super(SVRLSAC, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.gamma = torch.tensor(0.99, dtype=torch.float32, device=device)
        self.tau = tau
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.clip_grad_param = 5.0

        # entropy target as a tensor on device
        self.target_entropy = torch.tensor(-action_size, dtype=torch.float32, device=device)

        # separate (smaller) LR for alpha; safeguards during learn()
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # SVRL base params (unused directly, but kept)
        self.mask_prob = mask_prob
        self.lambda_struct = lambda_struct
        self.zeta = zeta

        # Actor
        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # Twin Q critics (+ targets)
        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)

        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        self.steps = 0

    # ---------------- API ----------------

    def get_action(self, state, eval=False):
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action = self.actor_local.get_det_action(state) if eval else self.actor_local.get_action(state)
        return action.cpu().numpy()

    # ---------------- Helpers ----------------

    def calc_policy_loss(self, states, alpha):
        """
        J_pi = E_s [ alpha * log_pi(a|s) - min(Q1(s,a), Q2(s,a)) ]
        """
        actions_pred, log_pis = self.actor_local.evaluate(states)  # (B,A), (B,1)
        q1 = self.critic1(states, actions_pred)                    # (B,1)
        q2 = self.critic2(states, actions_pred)                    # (B,1)
        min_Q = torch.min(q1, q2)
        actor_loss = (alpha.detach() * log_pis - min_Q).mean()
        return actor_loss, log_pis

    def _compute_policy_values(self, obs_pi, obs_q):
        actions_pred, log_pis = self.actor_local.evaluate(obs_pi)
        qs1 = self.critic1(obs_q, actions_pred)
        qs2 = self.critic2(obs_q, actions_pred)
        return qs1 - log_pis.detach(), qs2 - log_pis.detach()

    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = torch.log(torch.tensor(0.5, device=self.device)) * self.action_size
        return random_values - random_log_probs

    def _mask_prob_at(self, step: int) -> float:
        # E) slower decay, higher floor
        p0, pmin, tau = 0.40, 0.25, 5e5
        p = pmin + (p0 - pmin) * math.exp(- step / tau)
        return max(0.0, min(p, 0.9))

    def _lambda_at(self, step: int, alpha_val: float) -> float:
        # D) schedule lambda and link to alpha
        lam0, lam1, warm_steps = 0.3, 1.0, 1e3
        lam = lam0 + (lam1 - lam0) * min(1.0, step / warm_steps)
        # optional global cap if you want it:
        lam = min(lam, 1.0)
        return lam

    # ---------------- Core Learn ----------------

    def learn(self, experiences):
        """
        N×N shared-column Q-matrix:
          - columns are a_cols = π(next_states) shared across all rows
          - grid Q(s'_i, a'_j) computed with twin target critics, min over (Q1,Q2)
          - random masking via SVRL (mask_prob) + low-rank soft-impute(rank, zeta)
          - reconstructed diagonal used as structural target; mixed with true SAC target via λ(schedule)
        """
        states, actions, rewards, next_states, dones = experiences
        global_step = self.steps

        # ---------------- actor update ----------------
        # C) alpha safeguards & warmup
        if self.steps < 5000:
            with torch.no_grad():
                self.log_alpha.copy_(torch.tensor([-1.6094379], device=self.device))  # alpha ≈ 0.2
            current_alpha = self.log_alpha.exp().detach()
            actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
            alpha_loss = torch.zeros((), device=self.device)
        else:
            current_alpha = self.log_alpha.exp().detach()
            actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
            alpha_loss = -(self.log_alpha.exp() * (log_pis.detach() + self.target_entropy)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor_local.parameters(), self.clip_grad_param)
        self.actor_optimizer.step()

        if self.steps >= 5000:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-5.0, 2.0)

        self.alpha = self.log_alpha.exp().detach()

        # ---------------- critic target (N×N shared columns) ----------------
        with torch.no_grad():
            B = next_states.size(0)

            # shared columns: actions for each next state
            a_cols, new_log_pis = self.actor_local.evaluate(next_states)     # (B,A), (B,1)

            # build (B×B) grid
            S_grid = next_states.unsqueeze(1).expand(B, B, next_states.size(-1)).reshape(B * B, -1)  # (B*B,S)
            A_grid = a_cols.unsqueeze(0).expand(B, B, a_cols.size(-1)).reshape(B * B, -1)            # (B*B,A)

            # twin target critics on grid, then take min -> (B,B)
            q1_grid = self.critic1_target(S_grid, A_grid).squeeze(-1)  # (B*B,)
            q2_grid = self.critic2_target(S_grid, A_grid).squeeze(-1)  # (B*B,)
            qmin_grid = torch.min(q1_grid, q2_grid).view(B, B)         # (B,B)

            # A) normalize -> reconstruct in normalized space -> denormalize
            #q_mean = qmin_grid.mean()
            #q_std = qmin_grid.std().clamp_min(1e-6)
            #Qn = (qmin_grid - q_mean) / q_std

            # zeta in normalized space (either constant or spectral-proportional, then capped)
            #sval = torch.linalg.svdvals(Qn)
            #s1_n = sval.max()
            #zeta_n_eff = min(85.0, 0.4 * float(s1_n))   # ~0.2*s_max on normalized matrix, cap 3.0
            #zeta_n_eff = max(zeta_n_eff, 60.0)          # B) hard upper bound for safety
            zeta_n_eff = 75
            # E) masking schedule
            mp = 0.4

            Q_target_recon = softimp(qmin_grid, mask_prob=mp, zeta=zeta_n_eff, n_iter=100)
            #Q_target_recon = Qn_recon * q_std + q_mean                      # back to original scale (B,B)

            # reconstructed diagonal corresponds to (s'_i, a'_i)
            q_diag_recon = Q_target_recon.diag().unsqueeze(1)               # (B,1)

            # "true" SAC next Q: min(Q1,Q2) on (s', a_cols), minus alpha*log pi (clamped)
            q1t = self.critic1_target(next_states, a_cols)
            q2t = self.critic2_target(next_states, a_cols)
            ent_term = (self.alpha * new_log_pis)           # C) soft cap on entropy contribution
            q_true_next = torch.min(q1t, q2t) - ent_term                    # (B,1)

            # D) λ schedule & link to alpha
            #lam = self._lambda_at(global_step, float(self.alpha))
            lam = 1.0 
            # mix reconstructed and true next Q
            Q_target_next = (1.0 - lam) * q_true_next + lam * q_diag_recon  # (B,1)

            # full TD targets
            Q_targets = rewards + self.gamma * (1 - dones) * Q_target_next  # (B,1)

        # ---------------- critic losses ----------------
        q1 = self.critic1(states, actions)  # (B,1)
        q2 = self.critic2(states, actions)  # (B,1)

        critic1_loss = F.mse_loss(q1, Q_targets.detach())
        critic2_loss = F.mse_loss(q2, Q_targets.detach())

        # update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ---------------- target soft update ----------------
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        # ------------- lightweight debug metrics -------------
        dbg = {
            "alpha": float(self.alpha),
            "log_pi_mean": float(log_pis.mean().detach()),
            "log_pi_std": float(log_pis.std().detach()),
            "ent_term_max": float(ent_term.max().detach()),
            "mask_prob": float(mp),
            "zeta_n_eff": float(zeta_n_eff),
            "qgrid_std": float(qmin_grid.std().detach()),
            "qgrid_absmax": float(qmin_grid.abs().max().detach()),
            "qrecon_std": float(Q_target_recon.std().detach()),
            "lambda": float(lam),
            "q_true_next_mean": float(q_true_next.mean().detach()),
            "q_true_next_std": float(q_true_next.std().detach()),
            "Q_targets_mean": float(Q_targets.mean().detach()),
            "Q_targets_std": float(Q_targets.std().detach()),
        }

        self.steps += 1
        return (
            float(actor_loss.item()),
            float(alpha_loss.item()) if alpha_loss.numel() > 0 else 0.0,
            float(critic1_loss.item()),
            float(critic2_loss.item()),
            float(self.alpha.item()),
            zeta_n_eff,
            float(qmin_grid.std().item()),
            float(Q_target_recon.std().item())
        )

    # ---------------- utils ----------------

    def soft_update(self, local_model, target_model):
        """θ_target ← τ·θ_local + (1-τ)·θ_target"""
        for t_param, l_param in zip(target_model.parameters(), local_model.parameters()):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)


