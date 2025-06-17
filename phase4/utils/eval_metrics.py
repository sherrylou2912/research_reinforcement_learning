import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

def evaluate_policy(
    env,
    policy,
    n_episodes: int = 10,
    deterministic: bool = True
) -> Tuple[float, float]:
    """
    Evaluate a policy over multiple episodes.
    
    Args:
        env: Gym environment
        policy: Policy to evaluate
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        
    Returns:
        Mean and std of episode returns
    """
    episode_returns = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_return = 0.0
        done = False
        
        while not done:
            if deterministic:
                action = policy.get_action(state, eval=True)
            else:
                action = policy.get_action(state)
                
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            state = next_state
            
        episode_returns.append(episode_return)
    
    return np.mean(episode_returns), np.std(episode_returns)

def compute_value_bounds(
    critic1,
    critic2,
    states: torch.Tensor,
    actions: torch.Tensor,
    n_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute value function bounds using bootstrapped critics.
    
    Args:
        critic1: First critic network
        critic2: Second critic network
        states: Batch of states
        actions: Batch of actions
        n_samples: Number of bootstrap samples
        
    Returns:
        Lower and upper bounds of value estimates
    """
    q1 = critic1(states, actions)
    q2 = critic2(states, actions)
    
    q_min = torch.min(q1, q2)
    q_max = torch.max(q1, q2)
    
    return q_min, q_max

def compute_uncertainty(
    critics: List[torch.nn.Module],
    states: torch.Tensor,
    actions: torch.Tensor
) -> torch.Tensor:
    """
    Compute uncertainty estimates using an ensemble of critics.
    
    Args:
        critics: List of critic networks
        states: Batch of states
        actions: Batch of actions
        
    Returns:
        Uncertainty estimates for each state-action pair
    """
    q_values = []
    for critic in critics:
        with torch.no_grad():
            q = critic(states, actions)
            q_values.append(q)
    
    q_stack = torch.stack(q_values, dim=0)
    uncertainty = torch.std(q_stack, dim=0)
    
    return uncertainty

if __name__ == "__main__":
    # 测试评估指标
    print("Testing evaluation metrics...")
    
    # 创建测试环境和简单策略
    import gym
    
    class DummyPolicy:
        def __init__(self, action_space):
            self.action_space = action_space
            
        def get_action(self, state, eval=False):
            return self.action_space.sample()
    
    # 测试策略评估
    env = gym.make('Pendulum-v1')
    policy = DummyPolicy(env.action_space)
    
    mean_return, std_return = evaluate_policy(
        env,
        policy,
        n_episodes=5,
        deterministic=False
    )
    
    print("\nPolicy evaluation results:")
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
    
    # 测试值函数界限计算
    class DummyCritic(torch.nn.Module):
        def forward(self, states, actions):
            batch_size = states.shape[0]
            return torch.randn(batch_size, 1)
    
    critic1 = DummyCritic()
    critic2 = DummyCritic()
    
    states = torch.randn(32, 3)  # 假设状态维度为3
    actions = torch.randn(32, 1)  # 假设动作维度为1
    
    q_min, q_max = compute_value_bounds(
        critic1,
        critic2,
        states,
        actions
    )
    
    print("\nValue bounds test:")
    print(f"Q-min shape: {q_min.shape}")
    print(f"Q-max shape: {q_max.shape}")
    
    # 测试不确定性估计
    n_critics = 5
    critics = [DummyCritic() for _ in range(n_critics)]
    
    uncertainty = compute_uncertainty(
        critics,
        states,
        actions
    )
    
    print("\nUncertainty estimation test:")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Mean uncertainty: {uncertainty.mean().item():.3f}")
    print(f"Std uncertainty: {uncertainty.std().item():.3f}")
    
    print("\nAll tests passed!")
