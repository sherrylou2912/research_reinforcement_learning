import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple

from .sac import SAC
from .networks import update_target

class CQL(SAC):
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
        super().__init__(state_size, action_size, config)
        
        # CQL specific parameters
        self.cql_alpha = config.get('cql_alpha', 1.0)
        self.cql_tau = config.get('cql_tau', 10.0)
        self.cql_weight = config.get('cql_weight', 1.0)
        self.num_random = config.get('num_random', 10)
        self.with_lagrange = config.get('with_lagrange', False)
        self.target_action_gap = config.get('target_action_gap', 1.0)
        
        if self.with_lagrange:
            self.log_cql_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.cql_alpha_optimizer = torch.optim.Adam([self.log_cql_alpha], lr=self.learning_rate)
    
    def _compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CQL loss
        
        Args:
            states: Current states
            actions: Current actions
            next_states: Next states
            rewards: Rewards
            dones: Done flags
            
        Returns:
            CQL loss and metrics dictionary
        """
        batch_size = states.shape[0]
        
        # Get current Q-values
        q1_pred = self.critic1(states, actions)
        q2_pred = self.critic2(states, actions)
        
        # Sample random actions
        random_actions = torch.FloatTensor(
            batch_size * self.num_random, self.action_size
        ).uniform_(-1, 1).to(self.device)
        
        # Expand states to match random actions
        temp_states = states.unsqueeze(1).repeat(1, self.num_random, 1)
        temp_states = temp_states.view(-1, states.shape[-1])
        
        # Get Q-values for random actions
        q1_rand = self.critic1(temp_states, random_actions)
        q2_rand = self.critic2(temp_states, random_actions)
        
        # Sample actions from current policy
        policy_actions, log_probs = self.actor.evaluate(temp_states)
        q1_curr = self.critic1(temp_states, policy_actions)
        q2_curr = self.critic2(temp_states, policy_actions)
        
        # Compute CQL loss
        random_density = np.log(0.5 ** self.action_size)
        cat_q1 = torch.cat([
            q1_rand - random_density,
            q1_curr - log_probs.detach()
        ])
        cat_q2 = torch.cat([
            q2_rand - random_density,
            q2_curr - log_probs.detach()
        ])
        
        min_qf1_loss = torch.logsumexp(cat_q1 / self.cql_tau, dim=0) * self.cql_tau
        min_qf2_loss = torch.logsumexp(cat_q2 / self.cql_tau, dim=0) * self.cql_tau
        
        # Compute final CQL loss
        min_qf1_loss = min_qf1_loss.mean() - q1_pred.mean()
        min_qf2_loss = min_qf2_loss.mean() - q2_pred.mean()
        
        if self.with_lagrange:
            # Automatically adjust CQL weight using Lagrange multipliers
            cql_alpha = torch.clamp(self.log_cql_alpha.exp(), min=0.0, max=1000000.0)
            cql_loss = cql_alpha * (min_qf1_loss - self.target_action_gap)
            
            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = -(cql_loss.detach())
            cql_alpha_loss.backward()
            self.cql_alpha_optimizer.step()
        else:
            cql_loss = self.cql_weight * (min_qf1_loss + min_qf2_loss)
            cql_alpha = self.cql_weight
            cql_alpha_loss = 0
        
        return cql_loss, {
            'cql_loss': cql_loss.item(),
            'cql_q1_loss': min_qf1_loss.item(),
            'cql_q2_loss': min_qf2_loss.item(),
            'cql_alpha': cql_alpha.item() if self.with_lagrange else self.cql_weight,
            'cql_alpha_loss': cql_alpha_loss.item() if self.with_lagrange else 0
        }
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update network parameters
        
        Args:
            batch: Dictionary containing 'observations', 'actions', 'rewards',
                  'next_observations', 'terminals'
            
        Returns:
            Dictionary containing various loss values and metrics
        """
        # Get standard SAC losses
        metrics = super().update(batch)
        
        # Compute CQL loss
        cql_loss, cql_metrics = self._compute_cql_loss(
            batch['observations'].to(self.device),
            batch['actions'].to(self.device),
            batch['next_observations'].to(self.device),
            batch['rewards'].to(self.device),
            batch['terminals'].to(self.device)
        )
        
        # Update critics with CQL loss
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        cql_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        # Update metrics
        metrics.update(cql_metrics)
        return metrics
    
    def save(self, path: str):
        """
        Save model
        
        Args:
            path: Save path
        """
        checkpoint = super().save(path)
        if self.with_lagrange:
            checkpoint.update({
                'log_cql_alpha': self.log_cql_alpha,
                'cql_alpha_optimizer_state_dict': self.cql_alpha_optimizer.state_dict()
            })
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """
        Load model
        
        Args:
            path: Load path
        """
        checkpoint = torch.load(path, weights_only=False)
        super().load(path)
        if self.with_lagrange:
            self.log_cql_alpha = checkpoint['log_cql_alpha']
            self.cql_alpha_optimizer.load_state_dict(checkpoint['cql_alpha_optimizer_state_dict'])

if __name__ == "__main__":
    # Test code
    import gymnasium as gym
    
    # Create environment
    env = gym.make('Pendulum-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    # Create agent
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gamma': 0.99,
        'tau': 0.005,
        'learning_rate': 3e-4,
        'hidden_size': 256,
        'cql_alpha': 1.0,
        'cql_tau': 10.0,
        'with_lagrange': True,
        'target_action_gap': 1.0
    }
    agent = CQL(state_size, action_size, config)
    print("Agent created successfully!")
