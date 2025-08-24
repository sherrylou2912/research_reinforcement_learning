import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor
import numpy as np
import math
import copy
from svrl_utils import softimp


class SVRLSAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        tau,
                        hidden_size,
                        learning_rate,
                        n_action_sample,
                        rank,
                        mask_prob,
                        lambda_struct,
                        device
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(SVRLSAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = torch.FloatTensor([0.99]).to(device)
        self.tau = tau
        hidden_size = hidden_size
        learning_rate = learning_rate
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 

        # SVRL param
        self.n_action_sample = n_action_sample
        self.mask_prob = mask_prob
        self.lambda_struct = lambda_struct
        self.rank = rank 
        
        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 

    
    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
            else:
                action = self.actor_local.get_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states, actions_pred.squeeze(0))   
        q2 = self.critic2(states, actions_pred.squeeze(0))
        min_Q = torch.min(q1,q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q )).mean()
        return actor_loss, log_pis

    def _compute_policy_values(self, obs_pi, obs_q):
        #with torch.no_grad():
        actions_pred, log_pis = self.actor_local.evaluate(obs_pi)
        
        qs1 = self.critic1(obs_q, actions_pred)
        qs2 = self.critic2(obs_q, actions_pred)
        
        return qs1 - log_pis.detach(), qs2 - log_pis.detach()
    
    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs
    
    def learn(self, experiences):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        with torch.no_grad():
            B = next_states.size(0)
            K = self.n_action_sample
            A = self.action_size

            # Expand next_states for batch-wise K action sampling
            next_expanded = next_states.unsqueeze(1).repeat(1, K, 1).reshape(B * K, -1)
            a_pi_all, _ = self.actor_local.evaluate(next_expanded)
            _, new_log_pis = self.actor_local.evaluate(next_states) 

            # Compute target Q values
            q1 = self.critic1_target(next_expanded, a_pi_all)
            q2 = self.critic2_target(next_expanded, a_pi_all)
            qmin = torch.min(q1, q2).squeeze(-1).view(B, K)

            # SoftImpute-based low-rank reconstruction
            Q_target_recon = softimp(qmin, mask_prob=self.mask_prob, rank=self.rank, n_iter=30).to(self.device)

            # Align column corresponding to π(s') action
            a_pi_first = a_pi_all.view(B, K, A)[:, 0, :]  # [B, A]
            a_pi_all_full = a_pi_all.view(B, K, A)
            dists = torch.norm(a_pi_all_full - a_pi_first.unsqueeze(1), dim=2)
            align_idx = torch.argmin(dists, dim=1)
            row_idx = torch.arange(B, device=self.device)
            
            # Extract π(s') column Q-value from low-rank Q_target matrix
            Q_target_next_struct = Q_target_recon[row_idx, align_idx].unsqueeze(1)
            Q_target_next_true = torch.min(
                self.critic1_target(next_states, a_pi_first),
                self.critic2_target(next_states, a_pi_first)
            )- self.alpha.to(self.device) * new_log_pis

            Q_target_next = (1 - self.lambda_struct) * Q_target_next_true + self.lambda_struct * Q_target_next_struct
            Q_targets = rewards + self.gamma * (1 - dones) * (Q_target_next)

        # Compute critic loss
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, Q_targets)
        critic2_loss = F.mse_loss(q2, Q_targets)
        
        # Compute critic loss
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, Q_targets)
        critic2_loss = F.mse_loss(q2, Q_targets)
        
        total_c1_loss = critic1_loss
        total_c2_loss = critic2_loss
        
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
