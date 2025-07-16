import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import minari
from typing import Dict, Tuple, Optional
import gymnasium as gym

class MinariDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        normalize_states: bool = False,
        normalize_rewards: bool = False,
        download: bool = True
    ):
        """
        Initialize Minari dataset
        
        Args:
            dataset_name: Name of the Minari dataset
            normalize_states: Whether to normalize states
            normalize_rewards: Whether to normalize rewards
            download: Whether to download dataset if not found
        """
        # 加载数据集
        print(f"Loading Minari dataset: {dataset_name}")
        try:
            self.dataset = minari.load_dataset(dataset_name)
        except FileNotFoundError:
            if download:
                print(f"Dataset {dataset_name} not found locally. Downloading...")
                minari.download_dataset(dataset_name)
                self.dataset = minari.load_dataset(dataset_name)
            else:
                raise
        
        # 转换为numpy数组
        all_episodes = list(self.dataset.iterate_episodes())
        
        # 合并所有episode的数据
        self.observations = np.concatenate([ep.observations for ep in all_episodes])
        self.actions = np.concatenate([ep.actions for ep in all_episodes])
        self.rewards = np.concatenate([ep.rewards for ep in all_episodes])
        self.terminals = np.concatenate([ep.terminations for ep in all_episodes])
        
        # 构建 next_observations
        self.next_observations = []
        for ep in all_episodes:
            # 对于每个 episode，next_observations 是当前 observations 向后移一位
            # 最后一个 next_observation 使用零向量或当前状态
            ep_next_obs = np.roll(ep.observations, -1, axis=0)
            ep_next_obs[-1] = ep.observations[-1]  # 最后一个状态的下一个状态仍然是它自己
            self.next_observations.append(ep_next_obs)
        self.next_observations = np.concatenate(self.next_observations)
        
        # 确保所有数组长度一致
        min_length = min(len(self.observations), len(self.actions), 
                        len(self.rewards), len(self.terminals), len(self.next_observations))
        
        self.observations = self.observations[:min_length]
        self.actions = self.actions[:min_length]
        self.rewards = self.rewards[:min_length]
        self.terminals = self.terminals[:min_length]
        self.next_observations = self.next_observations[:min_length]
        
        # 计算归一化统计量
        self.state_mean = None
        self.state_std = None
        self.reward_mean = None
        self.reward_std = None
        
        if normalize_states:
            print("Computing state normalization statistics...")
            self.state_mean = np.mean(self.observations, axis=0)
            self.state_std = np.std(self.observations, axis=0) + 1e-8
            
            # 应用归一化
            self.observations = (self.observations - self.state_mean) / self.state_std
            self.next_observations = (self.next_observations - self.state_mean) / self.state_std
        
        if normalize_rewards:
            print("Computing reward normalization statistics...")
            self.reward_mean = np.mean(self.rewards)
            self.reward_std = np.std(self.rewards) + 1e-8
            
            # 应用归一化
            self.rewards = (self.rewards - self.reward_mean) / self.reward_std
        
        print(f"Dataset loaded with {len(self)} transitions")
        
    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # 检查索引是否在有效范围内
        if idx >= len(self.observations):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {len(self.observations)} samples")
        
        return {
            'observations': self.observations[idx].astype(np.float32),
            'actions': self.actions[idx].astype(np.float32),
            'rewards': self.rewards[idx].astype(np.float32),
            'next_observations': self.next_observations[idx].astype(np.float32),
            'terminals': self.terminals[idx].astype(np.float32)
        }

def create_minari_dataloader(
    dataset_name: str,
    batch_size: int,
    normalize_states: bool = False,
    normalize_rewards: bool = False,
    device: str = 'cuda',
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True
) -> Tuple[DataLoader, Optional[Dict]]:
    """
    创建Minari数据集的DataLoader
    
    Args:
        dataset_name: Minari数据集名称
        batch_size: 批量大小
        normalize_states: 是否归一化状态
        normalize_rewards: 是否归一化奖励
        device: 使用的设备
        num_workers: 数据加载的工作进程数
        pin_memory: 是否将数据固定在内存中
        download: 是否下载数据集（如果本地不存在）
        
    Returns:
        DataLoader和归一化统计量的元组
    """
    # 创建数据集
    dataset = MinariDataset(
        dataset_name=dataset_name,
        normalize_states=normalize_states,
        normalize_rewards=normalize_rewards,
        download=download
    )
    
    # 确保batch_size不会导致访问超出范围的索引
    total_samples = len(dataset)
    effective_batch_size = min(batch_size, total_samples // 2)  # 使用更保守的批处理大小
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=0,  # 禁用多进程加载
        pin_memory=pin_memory and device == 'cuda',
        drop_last=True
    )
    
    print(f"Dataset loaded with batch size {effective_batch_size}")
    
    # 返回归一化统计量
    normalization_stats = None
    if normalize_states or normalize_rewards:
        normalization_stats = {
            'state_mean': dataset.state_mean,
            'state_std': dataset.state_std,
            'reward_mean': dataset.reward_mean,
            'reward_std': dataset.reward_std
        }
    
    return data_loader, normalization_stats 