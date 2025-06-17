import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Union

try:
    from .networks import Actor, Critic, update_target
except ImportError:
    from networks import Actor, Critic, update_target

class SAC(nn.Module):
    """Soft Actor-Critic (SAC) algorithm implementation"""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Dict[str, Any]
    ):
        """
        Initialize SAC agent
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            config: Configuration dictionary containing hyperparameters
        """
        super(SAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.hidden_size = config.get('hidden_size', 256)
        
        # Initialize networks
        self.actor = Actor(state_size, action_size, self.hidden_size).to(self.device)
        self.critic1 = Critic(state_size, action_size, self.hidden_size).to(self.device)
        self.critic2 = Critic(state_size, action_size, self.hidden_size).to(self.device)
        self.critic1_target = Critic(state_size, action_size, self.hidden_size).to(self.device)
        self.critic2_target = Critic(state_size, action_size, self.hidden_size).to(self.device)
        
        # Copy parameters to target networks
        update_target(self.critic1_target, self.critic1, 1.0)
        update_target(self.critic2_target, self.critic2, 1.0)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.learning_rate)
        
        # Initialize temperature parameter alpha
        self.target_entropy = -action_size  # Target entropy is -|A|
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
        
        # Initialize normalization statistics
        self.normalization_stats = None
    
    def set_normalization_stats(self, stats: Dict[str, np.ndarray]):
        """
        Set normalization statistics for state and reward normalization
        
        Args:
            stats: Dictionary containing normalization statistics
        """
        self.normalization_stats = stats
    
    def normalize_state(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Normalize state using stored statistics
        
        Args:
            state: State to normalize
            
        Returns:
            Normalized state
        """
        if self.normalization_stats is None:
            return torch.as_tensor(state).float().to(self.device)
        return (state - self.normalization_stats['state_mean']) / self.normalization_stats['state_std']
    
    def get_action(self, state: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Select action using the current policy
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        state = self.normalize_state(state)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            action, _ = self.actor.evaluate(state)
        return action.cpu().numpy()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update network parameters
        
        Args:
            batch: Dictionary containing 'observations', 'actions', 'rewards',
                  'next_observations', 'terminals'
            
        Returns:
            Dictionary containing various loss values and metrics
        """
        # Move data to device
        states = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_observations'].to(self.device)
        dones = batch['terminals'].to(self.device)
        
        # Ensure rewards and dones have correct dimensions
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        
        # Compute target Q values
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.evaluate(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        q1_pred = self.critic1(states, actions)
        critic1_loss = nn.MSELoss()(q1_pred, q_target)
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        q2_pred = self.critic2(states, actions)
        critic2_loss = nn.MSELoss()(q2_pred, q_target)
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actions_pred, log_probs = self.actor.evaluate(states)
        q1_pred = self.critic1(states, actions_pred)
        q2_pred = self.critic2(states, actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)
        actor_loss = (self.log_alpha.exp() * log_probs - q_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        self.alpha_optimizer.zero_grad()
        _, new_log_probs = self.actor.evaluate(states)  # Get new log probs
        alpha_loss = -(self.log_alpha.exp() * (new_log_probs + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target networks
        update_target(self.critic1_target, self.critic1, self.tau)
        update_target(self.critic2_target, self.critic2, self.tau)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'q_value': q_pred.mean().item(),
            'batch_reward': rewards.mean().item()
        }
    
    def save(self, path: str):
        """
        Save model
        
        Args:
            path: Save path
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'normalization_stats': self.normalization_stats
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """
        Load model
        
        Args:
            path: Load path
        """
        checkpoint = torch.load(path, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.normalization_stats = checkpoint['normalization_stats']

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
        'hidden_size': 256
    }
    agent = SAC(state_size, action_size, config)
    print("Agent created successfully!")
