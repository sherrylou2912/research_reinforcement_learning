import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .networks import Critic, Actor 
import numpy as np
import math 
import copy
from typing import Dict, Any

class CQLSAC(nn.Module):
    """Conservative Q-Learning (CQL) algorithm implementation"""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Dict[str, Any]
    ):
        """
        Initialize CQL agent
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            config: Configuration dictionary containing hyperparameters
        """
        super(CQLSAC,self).__init__()

        self.device = config.get('device', 'cuda')
        # SAC Hyperparameter 
        self.tau = float(config.get('tau', 5e-3))
        self.hidden_size = int(config.get('hidden_size', 256))
        self.learning_rate = float(config.get('learning_rate', 3e-4))
        self.clip_grad_param = float(config.get('clip_grad_param', 1))
        self.target_entropy = float(config.get('target_entropy', -action_size))
        self.gamma = float(config.get('gamma', 0.99))
        
        # Actor parameters
        self.log_std_min = float(config.get('log_std_min', -20))
        self.log_std_max = float(config.get('log_std_max', 20))

        # Alpha is always tunable
        self.log_alpha = torch.tensor([0.0], requires_grad = True, device=self.device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params = [self.log_alpha], lr = self.learning_rate)

        # CQL specific parameters
        self.temp = config.get('temp', 1.0)
        self.with_lagrange = config.get('with_lagrange', False)
        self.cql_weight = config.get('cql_weight', 1.0)
        self.target_action_gap = config.get('target_action_gap', 10)
        self.cql_alpha = config.get('cql_alpha', 0.2)

        if self.with_lagrange:
            self.log_cql_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.cql_alpha_optimizer = torch.optim.Adam(params = [self.log_cql_alpha], lr=self.learning_rate)

        #rank evaluate parameter 
        self.num_random = config.get('num_random', 10)
        
        # Store action size for later use
        self.action_size = action_size
        
        # Initialize networks after all parameters are set
        self._init_networks(state_size, action_size)
        
    def _init_networks(self, state_size, action_size):
        """Initialize all neural networks and optimizers"""
        #Actor Network 
        self.actor_local = Actor(state_size, action_size, self.hidden_size, log_std_max= self.log_std_max, log_std_min = self.log_std_min, device=self.device)
        self.actor_local.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate)

        #Critic Network
        self.critic1 = Critic(state_size, action_size, self.hidden_size, self.device, 1229)
        self.critic1.to(self.device)
        self.critic2 = Critic(state_size, action_size, self.hidden_size, self.device, 1101)
        self.critic2.to(self.device)

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = Critic(state_size, action_size, self.hidden_size, self.device, 1229)
        self.critic1_target.to(self.device)
        self.critic2_target = Critic(state_size, action_size, self.hidden_size, self.device, 1101)
        self.critic2_target.to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.learning_rate)

    def get_action(self, state, eval = False):
        """ Returns actions for given state as per current policy. """

        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
            else:
                action = self.actor_local.get_action(state)

        return action.numpy()

    def _calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states, actions_pred.squeeze(0))
        q2 = self.critic2(states, actions_pred.squeeze(0))

        min_Q = torch.min(q1, q2)

        actor_loss = ((alpha * log_pis - min_Q)).mean()
        return actor_loss, log_pis
    

    def _compute_policy_value(self, obs_pi, obs_q):

        actions_pred , log_pis = self.actor_local.evaluate(obs_pi)

        qs1 = self.critic1(obs_q, actions_pred)
        qs2 = self.critic2(obs_q, actions_pred)

        return qs1 - log_pis.detach(), qs2 - log_pis.detach()

    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)

        return random_values - random_log_probs

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update agent using batch of experience - interface for train.py"""
        # Convert batch dict to tuple format expected by learn method and move to device
        states = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_observations'].to(self.device)
        dones = batch['terminals'].to(self.device)
        
        experience = (states, actions, rewards, next_states, dones)
        
        # Call learn method and return metrics dict
        actor_loss, alpha_loss, critic1_loss, critic2_loss, cql1_loss, cql2_loss, current_alpha, cql_alpha_loss, cql_alpha = self.learn(experience)
        
        return {
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss,
            'cql1_loss': cql1_loss,
            'cql2_loss': cql2_loss,
            'alpha': self.log_alpha.exp().item(),
            'cql_alpha_loss': cql_alpha_loss,
            'cql_alpha': cql_alpha
        }

    def learn(self, experience):
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

        states, actions, rewards, next_states, dones = experience 

        # ----------------------------- update actor ----------------------------- #
        actor_loss, log_pis = self._calc_policy_loss(states, self.alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #compute alpha loss 
        alpha_loss = - (self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()


        # ----------------------------- update critics ----------------------------- #
        with torch.no_grad():
            next_action, new_log_pi = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states, next_action)
            Q_target2_next = self.critic2_target(next_states, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha * new_log_pi
            Q_targets = rewards.unsqueeze(-1) + (self.gamma * (1 - dones).unsqueeze(-1) * Q_target_next)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, Q_targets)
        critic2_loss = F.mse_loss(q2, Q_targets)

        #CQL Addon 
        random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        num_repeat = int(random_actions.shape[0] / states.shape[0])
        temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])
        temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1).view(next_states.shape[0] * num_repeat, next_states.shape[1])
        
        current_pi_values1, current_pi_values2  = self._compute_policy_value(temp_states, temp_states)
        next_pi_values1, next_pi_values2 = self._compute_policy_value(temp_next_states, temp_states)
        
        random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(states.shape[0], num_repeat, 1)
        random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2).reshape(states.shape[0], num_repeat, 1)
        
        current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat, 1)
        current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat, 1)

        next_pi_values1 = next_pi_values1.reshape(states.shape[0], num_repeat, 1)
        next_pi_values2 = next_pi_values2.reshape(states.shape[0], num_repeat, 1)
        
        cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)
        
        assert cat_q1.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        assert cat_q2.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"
        

        cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q1.mean()) * self.cql_weight
        cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q2.mean()) * self.cql_weight
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.log_cql_alpha.exp(), min=0.0, max=1000000.0)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        else:
            # Use fixed CQL alpha when not using Lagrange multiplier
            cql_alpha = torch.FloatTensor([self.cql_alpha])
        
        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss
        
        
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
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), self.log_alpha.exp().item(), cql_alpha_loss.item(), cql_alpha.item()

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

    def set_normalization_stats(self, stats: Dict[str, np.ndarray]):
        """Set normalization statistics for states and rewards."""
        self.normalization_stats = stats
        if hasattr(self.actor_local, 'set_normalization_stats'):
            self.actor_local.set_normalization_stats(stats)

    def save(self, path: str):
        """Save agent state to file."""
        torch.save({
            'actor_state_dict': self.actor_local.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_cql_alpha': self.log_cql_alpha if self.with_lagrange else None,
            'cql_alpha_optimizer_state_dict': self.cql_alpha_optimizer.state_dict() if self.with_lagrange else None,
        }, path)

    def load(self, path: str):
        """Load agent state from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha'].data
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp().detach()
            
        if self.with_lagrange and checkpoint['log_cql_alpha'] is not None:
            self.log_cql_alpha.data = checkpoint['log_cql_alpha'].data
            self.cql_alpha_optimizer.load_state_dict(checkpoint['cql_alpha_optimizer_state_dict'])

