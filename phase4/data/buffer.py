import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from collections import deque
import random
from .minari_loader import MinariDataset

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size: int, batch_size: int, device: str):
        """Initialize a ReplayBuffer object.
        
        Args:
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
            device: device to store the tensors
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.normalization_stats = None
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """Add a new experience to memory."""
        self.memory.append({
            'observations': state,
            'actions': action,
            'rewards': reward,
            'next_observations': next_state,
            'terminals': done
        })

    def sample(self) -> Dict[str, torch.Tensor]:
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=min(self.batch_size, len(self.memory)))
        
        states = torch.FloatTensor(
            np.vstack([e['observations'] for e in experiences])
        ).to(self.device)
        actions = torch.FloatTensor(
            np.vstack([e['actions'] for e in experiences])
        ).to(self.device)
        rewards = torch.FloatTensor(
            np.vstack([e['rewards'] for e in experiences])
        ).to(self.device)
        next_states = torch.FloatTensor(
            np.vstack([e['next_observations'] for e in experiences])
        ).to(self.device)
        dones = torch.FloatTensor(
            np.vstack([e['terminals'] for e in experiences]).astype(np.uint8)
        ).to(self.device)

        return {
            'observations': states,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_states,
            'terminals': dones
        }
        
    def load_from_dataset(self, dataset: Union[MinariDataset, str], max_samples: Optional[int] = None):
        """从数据集加载数据到缓冲区
        
        Args:
            dataset: MinariDataset实例或数据集名称
            max_samples: 最大加载样本数，如果为None则加载全部
        """
        if isinstance(dataset, str):
            dataset = MinariDataset(
                dataset_name=dataset,
                device=self.device
            )
            
        # 保存数据集的标准化统计信息
        self.normalization_stats = dataset.get_normalization_stats()
        
        # 确定要加载的样本数
        num_samples = len(dataset)
        if max_samples is not None:
            num_samples = min(max_samples, num_samples)
            
        # 清空当前缓冲区
        self.memory.clear()
        
        # 加载数据
        indices = np.random.permutation(len(dataset))[:num_samples]
        for idx in indices:
            sample = dataset[idx]
            self.memory.append({
                'observations': sample['observations'].cpu().numpy(),
                'actions': sample['actions'].cpu().numpy(),
                'rewards': sample['rewards'].cpu().numpy(),
                'next_observations': sample['next_observations'].cpu().numpy(),
                'terminals': sample['terminals'].cpu().numpy()
            })
            
        print(f"Loaded {len(self.memory)} samples from dataset")
        
    def get_normalization_stats(self) -> Optional[Dict]:
        """获取数据集的标准化统计信息"""
        return self.normalization_stats

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)

if __name__ == "__main__":
    # 测试回放缓冲区
    print("Testing replay buffer...")
    
    # 创建回放缓冲区
    buffer_size = 1000
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    buffer = ReplayBuffer(buffer_size, batch_size, device)
    print(f"Created buffer with size {buffer_size}")
    
    # 测试从数据集加载
    try:
        print("\nTesting dataset loading...")
        buffer.load_from_dataset("D4RL/pen/expert-v2", max_samples=1000)
        
        # 获取标准化统计信息
        stats = buffer.get_normalization_stats()
        print("\nNormalization statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # 测试采样
        batch = buffer.sample()
        print("\nSample batch statistics:")
        for key, value in batch.items():
            print(f"{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Device: {value.device}")
            print(f"  Type: {value.dtype}")
            
    except Exception as e:
        print(f"\nError in dataset loading test: {str(e)}")
    
    # 测试手动添加样本
    print("\nTesting manual sample addition...")
    buffer = ReplayBuffer(buffer_size, batch_size, device)
    
    state_size = 4
    action_size = 2
    
    for i in range(100):
        state = np.random.randn(state_size)
        action = np.random.randn(action_size)
        reward = np.random.rand()
        next_state = np.random.randn(state_size)
        done = bool(np.random.rand() > 0.8)
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"Added {len(buffer)} transitions")
    
    # 测试采样
    batch = buffer.sample()
    print("\nSample batch statistics:")
    for key, value in batch.items():
        print(f"{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Device: {value.device}")
        print(f"  Type: {value.dtype}")
        
    # 验证批量大小
    assert all(v.shape[0] == batch_size for v in batch.values()), "Incorrect batch size!"
    print("\nAll tests passed!")
