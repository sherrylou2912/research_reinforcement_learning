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
        validate_dataset = False
    ):
        """
        Initialize Minari dataset
        
        Args:
            dataset_name: Name of the Minari dataset
        """
        # load dataset
        dataset = minari.load_dataset(dataset_name, download = True)
        print(f"loading dataset", dataset_name, "completed")
        self.env = dataset.recover_environment()
        print(f"recovering environment suceessful")

        # get dimensional information
        if len(dataset) == 0:
            raise ValueError(f"Dataset {dataset_name} is empty")
        
        first_ep = dataset[0]
        self.state_dim = first_ep.observations[0].shape[0]
        self.action_dim = first_ep.actions[0].shape[0]
        print("state_dim: ", self.state_dim, "action_dim: ", self.action_dim)

        # validate dataset compatibility
        if validate_dataset:
            self._validate_dataset(dataset)
        
        # store data
        self.obs, self.acts, self.rews, self.dones, self.next_obs = [],[],[],[],[]
        for ep in dataset:
            self._store_episode(
                ep.observations[:-1],
                ep.actions,
                ep.rewards,
                np.logical_or(ep.terminations, ep.truncations),
                ep.observations[1:]
            )

        # convert to numpy arrays for efficiency
        self.obs = np.array(self.obs)
        self.acts = np.array(self.acts)
        self.rews = np.array(self.rews)
        self.dones = np.array(self.dones)
        self.next_obs = np.array(self.next_obs)
        
        self._normalize()
        print('state, action data normalized, dataset normalization completed')
        self.priorities = np.ones(len(self.obs)) * 1e-5

    def _validate_dataset(self, dataset):
        """Validate dataset compatibility and consistency"""
        for i, ep in enumerate(dataset):
            # check episode has required attributes
            if not hasattr(ep, 'observations') or not hasattr(ep, 'actions'):
                raise ValueError(f"Episode {i} missing required attributes")
            
            # check observation/action shapes are consistent
            for j, obs in enumerate(ep.observations):
                if obs.shape[0] != self.state_dim:
                    raise ValueError(f"Episode {i}, step {j}: observation shape {obs.shape[0]} != expected {self.state_dim}")
            
            for j, act in enumerate(ep.actions):
                if act.shape[0] != self.action_dim:
                    raise ValueError(f"Episode {i}, step {j}: action shape {act.shape[0]} != expected {self.action_dim}")
            
            # check episode length consistency
            if len(ep.observations) != len(ep.actions) + 1:
                raise ValueError(f"Episode {i}: observations length {len(ep.observations)} != actions length + 1 {len(ep.actions) + 1}")
            
            if len(ep.actions) != len(ep.rewards):
                raise ValueError(f"Episode {i}: actions length {len(ep.actions)} != rewards length {len(ep.rewards)}")

    def _store_episode(self, obs, acts, rews, dones, next_obs):
        self.obs.extend(obs)
        self.acts.extend(acts)
        self.rews.extend(rews)
        self.dones.extend(dones)
        self.next_obs.extend(next_obs)

    def _normalize(self):
        # state normalization
        self.obs_mean = np.mean(self.obs, axis = 0)
        self.obs_std = np.std(self.obs, axis = 0) + 1e-8
        self.obs = (self.obs - self.obs_mean) / self.obs_std

        # action normalization - store stats but don't normalize actions
        self.act_mean = np.mean(self.acts, axis = 0)
        self.act_std = np.std(self.acts, axis = 0) + 1e-8
        print(f"🔍 DATASET ACTION ANALYSIS:")
        print(f"  Original action range: [{self.acts.min():.3f}, {self.acts.max():.3f}]")
        print(f"  Original action mean: {self.act_mean}")
        print(f"  Original action std: {self.act_std}")
        # Don't normalize actions - keep them in original range
        # self.acts = (self.acts - self.act_mean) / self.act_std

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = np.abs(priorities.flatten()) + 1e-5

    def get_normalization_stats(self):
        return {
            'state_mean': self.obs_mean,
            'state_std':self.obs_std,
            'action_mean':self.act_mean,
            'action_std': self.act_std
        }

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return (
            idx, 
            torch.FloatTensor(self.obs[idx]),
            torch.FloatTensor(self.acts[idx]),
            torch.FloatTensor(self.next_obs[idx]),
            torch.FloatTensor([self.rews[idx]]),
            torch.FloatTensor([bool(self.dones[idx])])
        )