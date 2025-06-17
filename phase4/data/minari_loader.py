import numpy as np
import torch
import gymnasium as gym
import minari
from typing import Dict, Tuple, Optional, List, Union
from torch.utils.data import Dataset, DataLoader

class MinariDataset(Dataset):
    """PyTorch Dataset class for loading and preprocessing Minari datasets"""
    
    def __init__(
        self,
        dataset_name: str,
        normalize_states: bool = True,
        normalize_rewards: bool = True,
        device: str = "cpu",
        download: bool = True
    ):
        """
        Initialize Minari dataset
        
        Args:
            dataset_name: Name of the Minari dataset
            normalize_states: Whether to normalize states
            normalize_rewards: Whether to normalize rewards
            device: Device to store data on ('cpu' or 'cuda')
            download: Whether to download dataset if not found locally
        """
        self.device = device
        try:
            self.dataset = minari.load_dataset(dataset_name)
        except FileNotFoundError:
            if download:
                print(f"Dataset {dataset_name} not found locally. Downloading...")
                minari.download_dataset(dataset_name)
                self.dataset = minari.load_dataset(dataset_name)
            else:
                raise
                
        # Extract data
        all_data = []
        for episode in self.dataset.iterate_episodes():
            observations = episode.observations
            next_observations = episode.observations[1:]  # Remove last state
            actions = episode.actions[:-1]  # Remove last action
            rewards = episode.rewards[:-1]  # Remove last reward
            terminals = np.zeros(len(actions))  # All steps are non-terminal except the last one
            terminals[-1] = episode.terminations[-1]
            
            # Create a sample for each timestep
            for t in range(len(actions)):
                all_data.append({
                    'observations': observations[t],
                    'actions': actions[t],
                    'rewards': rewards[t],
                    'next_observations': next_observations[t],
                    'terminals': terminals[t]
                })
        
        # Convert to numpy arrays
        self.observations = np.array([d['observations'] for d in all_data], dtype=np.float32)
        self.actions = np.array([d['actions'] for d in all_data], dtype=np.float32)
        self.rewards = np.array([d['rewards'] for d in all_data], dtype=np.float32)
        self.next_observations = np.array([d['next_observations'] for d in all_data], dtype=np.float32)
        self.terminals = np.array([d['terminals'] for d in all_data], dtype=np.float32)
        
        # Data normalization
        if normalize_states:
            self.state_mean = self.observations.mean(0, keepdims=True)
            self.state_std = self.observations.std(0, keepdims=True) + 1e-8
            
            self.observations = (self.observations - self.state_mean) / self.state_std
            self.next_observations = (self.next_observations - self.state_mean) / self.state_std
        else:
            self.state_mean = 0
            self.state_std = 1
            
        if normalize_rewards:
            self.reward_mean = self.rewards.mean()
            self.reward_std = self.rewards.std() + 1e-8
            self.rewards = (self.rewards - self.reward_mean) / self.reward_std
        else:
            self.reward_mean = 0
            self.reward_std = 1
            
        # Convert to torch tensors
        self.observations = torch.FloatTensor(self.observations).to(device)
        self.actions = torch.FloatTensor(self.actions).to(device)
        self.rewards = torch.FloatTensor(self.rewards).to(device)
        self.next_observations = torch.FloatTensor(self.next_observations).to(device)
        self.terminals = torch.FloatTensor(self.terminals).to(device)
        
    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'observations': self.observations[idx],
            'actions': self.actions[idx],
            'rewards': self.rewards[idx],
            'next_observations': self.next_observations[idx],
            'terminals': self.terminals[idx]
        }
    
    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Get normalization statistics"""
        return {
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std
        }
    
    def get_dataset_stats(self) -> Dict[str, Dict[str, float]]:
        """Get dataset statistics"""
        stats = {}
        for key, tensor in self.__getitem__(0).items():
            stats[key] = {
                'shape': list(tensor.shape),
                'mean': float(getattr(self, key).mean()),
                'std': float(getattr(self, key).std()),
                'min': float(getattr(self, key).min()),
                'max': float(getattr(self, key).max())
            }
        return stats

def create_minari_dataloader(
    dataset_name: str,
    batch_size: int = 256,
    normalize_states: bool = True,
    normalize_rewards: bool = True,
    device: str = "cpu",
    shuffle: bool = True,
    num_workers: int = 0,
    download: bool = True
) -> Tuple[DataLoader, Dict]:
    """
    Create Minari data loader
    
    Args:
        dataset_name: Name of the Minari dataset
        batch_size: Batch size
        normalize_states: Whether to normalize states
        normalize_rewards: Whether to normalize rewards
        device: Device to store data on
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        download: Whether to download dataset if not found
        
    Returns:
        data_loader: PyTorch data loader
        normalization_stats: Normalization statistics
    """
    dataset = MinariDataset(
        dataset_name=dataset_name,
        normalize_states=normalize_states,
        normalize_rewards=normalize_rewards,
        device=device,
        download=download
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    
    return data_loader, dataset.get_normalization_stats()

if __name__ == "__main__":
    # Test code
    print("Testing Minari dataset loading...")
    
    # Set test parameters
    dataset_name = "D4RL/pen/expert-v2"  # Use correct dataset name
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Create data loader
        data_loader, stats = create_minari_dataloader(
            dataset_name=dataset_name,
            batch_size=batch_size,
            device=device
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Device: {device}")
        print(f"Total samples: {len(data_loader.dataset)}")
        print(f"Number of batches: {len(data_loader)}")
        
        print("\nDataset statistics:")
        for key, value in stats.items():
            print(f"\n{key}:")
            for stat_name, stat_value in value.items():
                print(f"  {stat_name}: {stat_value}")
        
        # Test batch loading
        print("\nTesting batch loading...")
        batch = next(iter(data_loader))
        print("\nBatch information:")
        for key, value in batch.items():
            print(f"{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Device: {value.device}")
            print(f"  Type: {value.dtype}")
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"\nError occurred during testing:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}") 