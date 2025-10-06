import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob
        
    
    def get_action(self, state):
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
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CriticEnsemble(nn.Module):
    """Parallel Critic Ensemble with independent parameters for efficient GPU computation."""
    
    def __init__(self, state_size, action_size, hidden_size=32, n_critics=10, seed=1):
        """Initialize ensemble of independent critics for parallel computation.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_size (int): Number of nodes in the network layers
            n_critics (int): Number of critics in the ensemble
            seed (int): Random seed
        """
        super(CriticEnsemble, self).__init__()
        self.n_critics = n_critics
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Create independent layers for each critic
        # Each critic has its own parameters but we process them in parallel
        input_size = state_size + action_size
        
        # Layer 1: (input_size -> hidden_size) for each critic
        self.fc1_weights = nn.Parameter(torch.randn(n_critics, hidden_size, input_size))
        self.fc1_biases = nn.Parameter(torch.randn(n_critics, hidden_size))
        
        # Layer 2: (hidden_size -> hidden_size) for each critic  
        self.fc2_weights = nn.Parameter(torch.randn(n_critics, hidden_size, hidden_size))
        self.fc2_biases = nn.Parameter(torch.randn(n_critics, hidden_size))
        
        # Layer 3: (hidden_size -> 1) for each critic
        self.fc3_weights = nn.Parameter(torch.randn(n_critics, 1, hidden_size))
        self.fc3_biases = nn.Parameter(torch.randn(n_critics, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters following the same scheme as individual critics."""
        #torch.manual_seed(1)
        for i in range(self.n_critics):
            torch.manual_seed(i)  # 每个Critic不同的种子
            
            # 初始化第i个Critic的参数
            input_size = self.state_size + self.action_size
            lim1 = 1. / np.sqrt(input_size)
            self.fc1_weights.data[i].uniform_(-lim1, lim1)
            self.fc1_biases.data[i].uniform_(-lim1, lim1)
            
            lim2 = 1. / np.sqrt(self.hidden_size)
            self.fc2_weights.data[i].uniform_(-lim2, lim2)
            self.fc2_biases.data[i].uniform_(-lim2, lim2)
            
            self.fc3_weights.data[i].uniform_(-3e-3, 3e-3)
            self.fc3_biases.data[i].uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        """Forward pass through all critics in parallel.
        
        Args:
            state: Tensor of shape (batch_size, state_size)
            action: Tensor of shape (batch_size, action_size)
            
        Returns:
            Tensor of shape (n_critics, batch_size, 1)
        """
        batch_size = state.size(0)
        
        # Concatenate state and action
        x = torch.cat((state, action), dim=-1)  # (batch_size, input_size)
        
        # Expand input for all critics
        x = x.unsqueeze(0).expand(self.n_critics, -1, -1)  # (n_critics, batch_size, input_size)
        
        # First layer: parallel matrix multiplication for all critics
        # x: (n_critics, batch_size, input_size)
        # fc1_weights: (n_critics, hidden_size, input_size)
        # Result: (n_critics, batch_size, hidden_size)
        x = torch.bmm(x, self.fc1_weights.transpose(-2, -1)) + self.fc1_biases.unsqueeze(1)
        x = F.relu(x)
        
        # Second layer: parallel matrix multiplication for all critics
        # x: (n_critics, batch_size, hidden_size)
        # fc2_weights: (n_critics, hidden_size, hidden_size)
        # Result: (n_critics, batch_size, hidden_size)
        x = torch.bmm(x, self.fc2_weights.transpose(-2, -1)) + self.fc2_biases.unsqueeze(1)
        x = F.relu(x)
        
        # Third layer: parallel matrix multiplication for all critics
        # x: (n_critics, batch_size, hidden_size)
        # fc3_weights: (n_critics, 1, hidden_size)
        # Result: (n_critics, batch_size, 1)
        x = torch.bmm(x, self.fc3_weights.transpose(-2, -1)) + self.fc3_biases.unsqueeze(1)
        
        return x  # (n_critics, batch_size, 1)
    
    def get_individual_critic(self, critic_idx):
        """Get a view of a specific critic to maintain compatibility."""
        class IndividualCritic(nn.Module):
            def __init__(self, ensemble, idx):
                super().__init__()
                self.ensemble = ensemble
                self.idx = idx
            
            def forward(self, state, action):
                all_outputs = self.ensemble(state, action)
                return all_outputs[self.idx]
            
            def parameters(self):
                # Return parameters for this specific critic
                for param in self.ensemble.parameters():
                    yield param
        
        return IndividualCritic(self, critic_idx)