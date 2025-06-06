import numpy as np
import gym as gym
from tqdm import tqdm
import random
import rl_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
    
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample).sum(dim=-1, keepdim=True)  # ⬅️ sum over action_dim
        action = torch.tanh(normal_sample)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7).sum(dim=-1, keepdim=True)  # ⬅️ sum again
        action = action * self.action_bound
    
        return action, log_prob

class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, 
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device):
        self.action_dim = action_dim
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.actor(state)[0]
        return action.squeeze().detach().cpu().numpy() 

    
    def calc_target(self, rewards, next_state, dones):
        next_action, log_prob = self.actor(next_state)
        entropy = - log_prob
        q1_value = self.target_critic1(next_state, next_action)
        q2_value = self.target_critic2(next_state, next_action)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.from_numpy(np.array(transition_dict['actions'])).float().to(self.device)
        actions = actions.view(-1, self.action_dim)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        #rewards = (rewards + 8.0) / 8.0

        # Update critic
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(F.mse_loss(self.critic1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic2(states, actions), td_target.detach()))
        self.critic1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor 
        new_action, log_prob = self.actor(states)
        entropy = - log_prob
        q1_value = self.critic1(states, new_action)
        q2_value = self.critic2(states, new_action)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha 
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

class CQL_SAC:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, 
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device, beta, num_random):
        self.action_dim = action_dim
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau

        self.beta = beta
        self.num_random = num_random
        self.device = device

    def take_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.actor(state)[0]
        return action.squeeze().detach().cpu().numpy()

    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,self.action_dim).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        #rewards = (rewards + 8.0) / 8.0

        next_actions, log_prob = self.actor(next_states)
        entropy = - log_prob
        q1_value = self.critic1(next_states, next_actions)
        q2_value = self.critic2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        td_target = torch.clamp(td_target, min=-100, max=100)

        critic_1_loss = torch.mean(
            F.mse_loss(self.critic1(states, actions), td_target.detach()) 
        )
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic2(states, actions), td_target.detach()) 
        )

        # CQL update context (different from SAC)
        batch_size = states.shape[0]
        random_unif_actions = torch.rand(
            [batch_size * self.num_random, actions.shape[-1]], dtype=torch.float).uniform_(-1,1).to(self.device)
        random_unif_log_pi = torch.tensor(
                np.log(0.5 ** next_actions.shape[-1]),
                dtype=torch.float32,
                device=self.device
            )
        tmp_states = states.unsqueeze(1).repeat(1, self.num_random,1).view(-1, states.shape[-1])
        tmp_next_states = next_states.unsqueeze(1).repeat(1, self.num_random,1).view(-1, states.shape[-1])
        random_curr_actions, random_curr_log_pi = self.actor(tmp_states)
        random_next_actions, random_next_log_pi = self.actor(tmp_next_states)
        q1_unif = self.critic1(tmp_states, random_unif_actions).view(-1, self.num_random, 1)
        q2_unif = self.critic2(tmp_states, random_unif_actions).view(-1, self.num_random, 1)
        q1_curr = self.critic1(tmp_states, random_curr_actions).view(-1, self.num_random, 1)
        q2_curr = self.critic2(tmp_states, random_curr_actions).view(-1, self.num_random, 1)
        q1_next = self.critic1(tmp_states, random_next_actions).view(-1, self.num_random, 1)
        q2_next = self.critic2(tmp_states, random_next_actions).view(-1, self.num_random, 1)
        q1_cat = torch.cat([
            q1_unif - random_unif_log_pi,
            q1_curr - random_curr_log_pi.detach().view(-1, self.num_random, 1),
            q1_next - random_next_log_pi.detach().view(-1, self.num_random, 1)
        ], dim = 1)
        q2_cat = torch.cat([
            q2_unif - random_unif_log_pi,
            q2_curr - random_curr_log_pi.detach().view(-1, self.num_random, 1),
            q2_next - random_next_log_pi.detach().view(-1, self.num_random, 1)
        ], dim = 1)

        max_q1 = q1_cat.max(dim=1, keepdim=True).values
        qf1_loss_1 = (max_q1 + torch.log(torch.exp(q1_cat - max_q1).sum(dim=1, keepdim=True))).mean()
        max_q2 = q2_cat.max(dim=1, keepdim=True).values
        qf2_loss_1 = (max_q2 + torch.log(torch.exp(q2_cat - max_q2).sum(dim=1, keepdim=True))).mean()

        qf1_loss_2 = self.critic1(states, actions).mean()
        qf2_loss_2 = self.critic2(states, actions).mean()
        penalty1 = torch.clamp(qf1_loss_1 - qf1_loss_2, min=0, max=30)
        penalty2 = torch.clamp(qf2_loss_1 - qf2_loss_2, min=0, max=30)
        qf1_loss = critic_1_loss + self.beta * penalty1
        qf2_loss = critic_2_loss + self.beta * penalty2

        if torch.isnan(qf1_loss) or torch.isnan(qf2_loss):
            print("💥 NaN detected in critic loss!")
        if torch.isinf(qf1_loss) or torch.isinf(qf2_loss):
            print("💥 Inf detected in critic loss!")

        self.critic1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=50.0)
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=50.0)
        self.critic2_optimizer.step()

        # updat policy network
        new_actions, log_prob = self.actor(states)
        entropy = - log_prob
        q1_value = self.critic1(states, new_actions)
        q2_value = self.critic2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update alpha
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)


        

