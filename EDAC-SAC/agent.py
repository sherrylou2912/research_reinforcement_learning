import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor, CriticEnsemble
import numpy as np
import math
import copy
from svrl_utils import softimp


class EDACSAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,  #sac parameters 
                        state_size,
                        action_size,
                        tau,
                        hidden_size,
                        learning_rate,
                        # edac parameters 
                        n_q_networks,
                        eta,
                        #software parameters 
                        device,
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(EDACSAC, self).__init__()
        #SAC parameter 
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = torch.FloatTensor([0.99]).to(device)
        self.tau = tau
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.clip_grad_param = 1

        self.target_entropy = torch.tensor(-action_size, dtype=torch.float32, device=device)  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 

        #edac parameter 
        self.eta = eta 
        self.n_q_networks = n_q_networks 
        
        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network Ensemble (w/ Target Network) - Parallel Implementation
        self.critic_ensemble = CriticEnsemble(state_size, action_size, hidden_size, n_q_networks).to(self.device)
        self.critic_ensemble_target = CriticEnsemble(state_size, action_size, hidden_size, n_q_networks).to(self.device)
        
        # Copy parameters to target network
        self.critic_ensemble_target.load_state_dict(self.critic_ensemble.state_dict())
        
        # Single optimizer for the entire ensemble
        self.critics_optimizer = optim.Adam(self.critic_ensemble.parameters(), lr=self.learning_rate)
        
        # Backward compatibility - create individual critic views
        self.critics = [self.critic_ensemble.get_individual_critic(i) for i in range(n_q_networks)]
        self.critics_target = [self.critic_ensemble_target.get_individual_critic(i) for i in range(n_q_networks)]
        self.critics_optimizers = [self.critics_optimizer for _ in range(n_q_networks)]  # All point to same optimizer
        
        # Backward compatibility for rank.py module
        self.critic1 = self.critics[0]
        self.critic2 = self.critics[1] if n_q_networks > 1 else self.critics[0]

    
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

        # Use parallel ensemble computation
        q_values = self.critic_ensemble(states, actions_pred.squeeze(0))  # (n_critics, batch_size, 1)
        q_values = q_values.squeeze(-1).transpose(0, 1)  # (batch_size, n_critics)

        min_Q = torch.min(q_values, dim=1)[0]
        actor_loss = ((alpha * log_pis - min_Q)).mean()
        return actor_loss, log_pis


    def _compute_policy_values(self, obs_pi, obs_q):
        #with torch.no_grad():
        actions_pred, log_pis = self.actor_local.evaluate(obs_pi)
        
        # Use parallel ensemble computation
        q_values = self.critic_ensemble(obs_q, actions_pred)  # (n_critics, batch_size, 1)
        qs = [q_values[i] for i in range(self.n_q_networks)]
        
        return [q - log_pis.detach() for q in qs]
    
    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs
    
    def _compute_ensemble_similarity(self, states, actions):
        if self.eta == 0.0:
            return torch.tensor(0.0, device=self.device)
        
        # Create actions that require gradients
        actions_for_grad = actions.detach().clone().requires_grad_(True)
        
        # Compute Q-values for all critics in parallel
        q_values = self.critic_ensemble(states, actions_for_grad)  # (n_critics, batch_size, 1)
        
        gradients = []
        
        # Compute gradients for each critic
        for i in range(self.n_q_networks):
            grad = torch.autograd.grad(
                outputs=q_values[i].sum(),
                inputs=actions_for_grad,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if grad is not None:
                gradients.append(grad)

        if len(gradients) < 2:
            return torch.tensor(0.0, device=self.device)

        # Compute similarity between gradients
        similarity_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        count = 0
        
        for i in range(len(gradients)):
            for j in range(i + 1, len(gradients)):
                grad_i_norm = F.normalize(gradients[i], dim=-1, eps=1e-8)
                grad_j_norm = F.normalize(gradients[j], dim=-1, eps=1e-8)
                similarity = (grad_i_norm * grad_j_norm).sum(dim=-1).mean()
                similarity_loss = similarity_loss + similarity
                count += 1

        return similarity_loss / count if count > 0 else torch.tensor(0.0, device=self.device)
    
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
        alpha_loss = - (self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        with torch.no_grad():
            next_actions, next_log_probs = self.actor_local.evaluate(next_states)
            
            # Get target Q-values from all target critics in parallel
            target_q_values = self.critic_ensemble_target(next_states, next_actions)  # (n_critics, batch_size, 1)
            target_q_values = target_q_values.squeeze(-1)  # (n_critics, batch_size)
            
            Q_target_next = torch.min(target_q_values, dim=0)[0]
            Q_targets = rewards.squeeze(-1) + self.gamma * (1 - dones).squeeze(-1) * (Q_target_next - self.alpha.to(self.device) * next_log_probs.squeeze(-1))

        # Compute ensemble similarity loss
        ensemble_sim_loss = self._compute_ensemble_similarity(states, actions)
        
        # Compute Q-values for all critics in parallel
        q_values = self.critic_ensemble(states, actions)  # (n_critics, batch_size, 1)
        q_values = q_values.squeeze(-1)  # (n_critics, batch_size)
        
        # Compute critic losses for all critics
        critic_losses = []
        total_critic_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for i in range(self.n_q_networks):
            critic_loss = F.mse_loss(q_values[i], Q_targets)
            critic_losses.append(critic_loss.item())
            total_critic_loss = total_critic_loss + critic_loss
        
        # Add ensemble similarity loss
        total_loss = total_critic_loss + self.eta * ensemble_sim_loss
        
        # Update all critics together
        self.critics_optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(self.critic_ensemble.parameters(), self.clip_grad_param)
        self.critics_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_ensemble, self.critic_ensemble_target)
        
        avg_critic_loss = np.mean(critic_losses)
        return actor_loss.item(), alpha_loss.item(), avg_critic_loss, avg_critic_loss, current_alpha

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
