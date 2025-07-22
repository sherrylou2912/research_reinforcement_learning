import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Union, Dict

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1./np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        init_w = 3e-3, 
        log_std_min: float = -2,
        log_std_max: float = 2,
        device: str = "cpu"
    ):
        super(Actor,self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

        self.device = device

        print(" Before reset_parameters")
        print(f" mu.weight range: {self.mu.weight.data.min():.2f} to {self.mu.weight.data.max():.2f}")
        self.reset_parameters()
        print(" After reset_parameters")
        print(f" mu.weight range: {self.mu.weight.data.min():.2f} to {self.mu.weight.data.max():.2f}")
    
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the state and return the action and log probability"""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        return action, log_prob
    
    def get_action(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Get deterministic action (for evaluation)."""
        mu, log_std = self.forward(state)

        #print(f' Actor DEBUG:')
        #print(f' raw mu range: {mu.min():.2f} to {mu.max():.2f}')
        #print(f" raw mu sample: {mu.flatten()[:3]}")

        action = torch.tanh(mu)
        #print(f" Tanh action range: {action.min():.2f} to {action.max():.2f}")
        #print(f" Tanh action sample: {action.flatten()[:3]}")

        return torch.tanh(mu).detach().cpu()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

        self.mu.weight.data.uniform_(-0.003, 0.003)
        self.mu.bias.data.uniform_(-0.003, 0.003)

        self.log_std_linear.weight.data.uniform_(-0.1, 0.1)
        self.log_std_linear.bias.data.uniform_(-0.5,0.5)


class Critic(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        device: str = "cpu",
        seed = 1229
    ):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.reset_parameters()
        self.device = device

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-2, 3e-2)
        self.fc4.bias.data.fill_(0)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return torch.clamp(q_value, min = -50, max = 50)
        #return q_value


def update_target(target_net: nn.Module, local_net: nn.Module, tau: float):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    
    Args:
        target_net: Target network to update
        local_net: Local network to copy from
        tau: Interpolation parameter (usually small, e.g. 0.005)
    """
    for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)