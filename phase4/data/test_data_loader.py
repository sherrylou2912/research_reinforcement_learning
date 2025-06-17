import numpy as np
import torch
from typing import Dict

def create_test_dataset(
    num_samples: int = 1000,
    state_dim: int = 3,
    action_dim: int = 2
) -> Dict[str, np.ndarray]:
    """创建一个测试数据集"""
    return {
        'observations': np.random.randn(num_samples, state_dim),
        'actions': np.random.uniform(-1, 1, (num_samples, action_dim)),
        'rewards': np.random.randn(num_samples, 1),
        'next_observations': np.random.randn(num_samples, state_dim),
        'terminals': np.zeros((num_samples, 1))
    }

class TestDataset(torch.utils.data.Dataset):
    """测试用数据集类"""
    
    def __init__(
        self,
        num_samples: int = 1000,
        state_dim: int = 3,
        action_dim: int = 2,
        normalize_states: bool = True,
        normalize_rewards: bool = True
    ):
        self.dataset = create_test_dataset(num_samples, state_dim, action_dim)
        
        # 标准化状态
        if normalize_states:
            state_mean = self.dataset['observations'].mean(0)
            state_std = self.dataset['observations'].std(0) + 1e-3
            
            self.dataset['observations'] = (self.dataset['observations'] - state_mean) / state_std
            self.dataset['next_observations'] = (self.dataset['next_observations'] - state_mean) / state_std
        
        # 标准化奖励
        if normalize_rewards:
            reward_mean = self.dataset['rewards'].mean()
            reward_std = self.dataset['rewards'].std() + 1e-3
            
            self.dataset['rewards'] = (self.dataset['rewards'] - reward_mean) / reward_std
    
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
    print("Testing data loader functionality...")
    
    # 创建测试数据集
    dataset = create_test_dataset()
    
    print("\nRaw dataset statistics:")
    for key, value in dataset.items():
        print(f"{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Type: {value.dtype}")
        print(f"  Mean: {value.mean():.3f}")
        print(f"  Std: {value.std():.3f}")
    
    # 测试PyTorch数据集
    torch_dataset = TestDataset()
    
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