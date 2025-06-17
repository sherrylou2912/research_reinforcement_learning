import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from .networks import Critic, Actor
import numpy as np
import math
import copy
from completion.soft_impute import softimp
from typing import Dict, Any, Optional, Union, Tuple

from .sac import SAC
from .networks import update_target

try:
    from .cql import CQL
    from .networks import update_target
except ImportError:
    from cql import CQL
    from networks import update_target


class SVRL(CQL):
    """Structured Value-based Representation Learning (SVRL) algorithm implementation"""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Dict[str, Any]
    ):
        """
        Initialize SVRL agent
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            config: Configuration dictionary containing hyperparameters
        """
        super().__init__(state_size, action_size, config)
        
        # SVRL specific parameters
        self.rank = config.get('rank', 10)
        self.svrl_weight = config.get('svrl_weight', 1.0)
        self.soft_impute_iters = config.get('soft_impute_iters', 100)
        self.soft_impute_threshold = config.get('soft_impute_threshold', 1e-4)
        
        # Initialize low-rank approximation matrices
        self.U = nn.Parameter(torch.randn(state_size, self.rank).to(self.device))
        self.V = nn.Parameter(torch.randn(action_size, self.rank).to(self.device))
        
        # Initialize optimizer for low-rank matrices
        self.svrl_optimizer = torch.optim.Adam(
            [self.U, self.V],
            lr=self.learning_rate
        )
    
    def _compute_structured_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute structured Q-value using low-rank approximation
        
        Args:
            states: Current states
            actions: Current actions
            
        Returns:
            Structured Q-values
        """
        # Project states and actions to low-rank space
        state_proj = torch.matmul(states, self.U)
        action_proj = torch.matmul(actions, self.V)
        
        # Compute structured Q-value
        q_struct = torch.sum(state_proj * action_proj, dim=-1, keepdim=True)
        return q_struct
    
    def _compute_svrl_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q_values: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SVRL loss
        
        Args:
            states: Current states
            actions: Current actions
            q_values: Q-values from critic networks
            
        Returns:
            SVRL loss and metrics dictionary
        """
        # Compute structured Q-values
        q_struct = self._compute_structured_q(states, actions)
        
        # Compute loss between structured and original Q-values
        svrl_loss = nn.MSELoss()(q_struct, q_values)
        
        # Add regularization for low-rank matrices
        reg_loss = (torch.norm(self.U) + torch.norm(self.V)) * 0.01
        total_loss = self.svrl_weight * (svrl_loss + reg_loss)
        
        return total_loss, {
            'svrl_loss': svrl_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_svrl_loss': total_loss.item()
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
        # Get CQL losses and metrics
        metrics = super().update(batch)
        
        # Get Q-values for SVRL
        states = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        with torch.no_grad():
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)
            q_values = torch.min(q1, q2)
        
        # Compute and update SVRL loss
        svrl_loss, svrl_metrics = self._compute_svrl_loss(states, actions, q_values)
        
        self.svrl_optimizer.zero_grad()
        svrl_loss.backward()
        self.svrl_optimizer.step()
        
        # Update metrics
        metrics.update(svrl_metrics)
        return metrics
    
    def save(self, path: str):
        """
        Save model
        
        Args:
            path: Save path
        """
        checkpoint = super().save(path)
        checkpoint.update({
            'U': self.U.data,
            'V': self.V.data,
            'svrl_optimizer_state_dict': self.svrl_optimizer.state_dict()
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
        self.U.data = checkpoint['U']
        self.V.data = checkpoint['V']
        self.svrl_optimizer.load_state_dict(checkpoint['svrl_optimizer_state_dict'])

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
        'target_action_gap': 1.0,
        'rank': 10,
        'svrl_weight': 1.0,
        'soft_impute_iters': 100,
        'soft_impute_threshold': 1e-4
    }
    agent = SVRL(state_size, action_size, config)
    print("Agent created successfully!")
