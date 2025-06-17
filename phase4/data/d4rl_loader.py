import numpy as np
import torch
import gymnasium as gym
import minari
from typing import Dict, Tuple, Optional

def load_d4rl_dataset(
    env_name: str,
    normalize_states: bool = True,
    normalize_rewards: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load and preprocess a D4RL dataset using d4rl_new and minari.
    
    Args:
        env_name: Name of the D4RL environment
        normalize_states: Whether to normalize states
        normalize_rewards: Whether to normalize rewards
        
    Returns:
        Dictionary containing the preprocessed dataset
    """
    # Load environment and dataset
    env = gym.make(env_name)
    dataset = minari.load_dataset(env_name)
    
    # Extract components
    observations = dataset.observations
    actions = dataset.actions
    rewards = dataset.rewards
    next_observations = dataset.next_observations
    terminals = dataset.terminations
    
    # Convert to numpy arrays
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    terminals = np.array(terminals)
    
    # Normalize states if requested
    if normalize_states:
        state_mean = observations.mean(0)
        state_std = observations.std(0) + 1e-3
        
        observations = (observations - state_mean) / state_std
        next_observations = (next_observations - state_mean) / state_std
    
    # Normalize rewards if requested
    if normalize_rewards:
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-3
        
        rewards = (rewards - reward_mean) / reward_std
    
    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'next_observations': next_observations,
        'terminals': terminals
    }

class D4RLDataset(torch.utils.data.Dataset):
    """PyTorch dataset for D4RL data."""
    
    def __init__(
        self,
        env_name: str,
        normalize_states: bool = True,
        normalize_rewards: bool = True
    ):
        self.dataset = load_d4rl_dataset(
            env_name,
            normalize_states,
            normalize_rewards
        )
        
    def __len__(self) -> int:
        return len(self.dataset['observations'])
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            'observations': self.dataset['observations'][idx],
            'actions': self.dataset['actions'][idx],
            'rewards': self.dataset['rewards'][idx],
            'next_observations': self.dataset['next_observations'][idx],
            'terminals': self.dataset['terminals'][idx]
        }

if __name__ == "__main__":
    # 测试D4RL数据集加载
    print("Testing D4RL dataset loading...")
    
    # 测试环境
    env_name = "pen-expert-v1"  # 使用新的环境名称格式
    
    # 测试数据集加载
    dataset = load_d4rl_dataset(
        env_name,
        normalize_states=True,
        normalize_rewards=True
    )
    
    print("\nDataset statistics:")
    for key, value in dataset.items():
        print(f"{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Type: {value.dtype}")
        print(f"  Mean: {value.mean():.3f}")
        print(f"  Std: {value.std():.3f}")
    
    # 测试PyTorch数据集
    torch_dataset = D4RLDataset(
        env_name,
        normalize_states=True,
        normalize_rewards=True
    )
    
    print(f"\nDataset size: {len(torch_dataset)}")
    
    # 测试数据加载
    sample = torch_dataset[0]
    print("\nSample from dataset:")
    for key, value in sample.items():
        print(f"{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Type: {type(value)}")
    
    # 测试数据加载器
    dataloader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=32,
        shuffle=True
    )
    
    batch = next(iter(dataloader))
    print("\nBatch statistics:")
    for key, value in batch.items():
        print(f"{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Type: {type(value)}")
    
    print("\nAll tests passed!")
