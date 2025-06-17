import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Union, Dict

class Actor(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
        device: str = "cpu"
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device
        
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)
        
        self.to(device)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估状态并返回动作和对数概率"""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample()
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        return action, log_prob
    
    def get_action(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """获取随机动作"""
        # 确保状态是正确的格式和设备
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            mu, log_std = self.forward(state)
            std = log_std.exp()
            dist = Normal(mu, std)
            e = dist.rsample()
            action = torch.tanh(e)
        return action
    
    def get_det_action(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """获取确定性动作（用于评估）"""
        # 确保状态是正确的格式和设备
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            mu, _ = self.forward(state)
            action = torch.tanh(mu)
        return action

class Critic(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        
        self.net = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.to(device)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """评估状态-动作对的值"""
        # 确保输入是正确的格式
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
    
    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """评估一个批次的数据"""
        return self.forward(batch['observations'], batch['actions'])

def update_target(target: nn.Module, source: nn.Module, tau: float = 1.0):
    """软更新目标网络参数"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
