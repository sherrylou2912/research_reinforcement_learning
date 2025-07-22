
import os
import gymnasium as gym
# import d4rl  # 移除d4rl
import torch
import yaml
import argparse
from typing import Dict, Any
import numpy as np
import time # Added for timing
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import deque
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
import wandb
import glob

from agent.sac import SAC
from agent.cqlsac import CQLSAC
from agent.svrl import SVRL
from data.minari_loader import MinariDataset  # 用minari数据加载
from utils.logger import Logger
from utils.eval_metrics import evaluate_policy
from utils.rank_utils import log_approximate_rank

class Trainer:
    def __init__(self, dataset_name, agent_type, config_path):
        self.dataset_name = dataset_name
        self.dataset = MinariDataset(dataset_name)
        self.state_dim = self.dataset.state_dim
        self.action_dim = self.dataset.action_dim
        self.agent_type = agent_type
        self.config = self.load_config(config_path)

    def load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config 
    
    def create_agent(self):
        if self.agent_type == 'sac':
            return SAC(self.state_dim, self.action_dim, self.config)
        elif self.agent_type == 'cqlsac':
            return CQLSAC(self.state_dim, self.action_dim, self.config)
        elif self.agent_type == 'svrl':
            return SVRL(self.state_dim, self.action_dim, self.config)
        else: 
            raise ValueError(f"Unknown agent type: {self.agent_type}")
        
    def evaluate(self, env, policy):
        print(f"\n🎮 EVALUATION DEBUG:")
        reward_batch = []
        eval_runs = self.config.get('eval_episodes', 5)
        
        for i in range(eval_runs):
            print(f"\n--- Evaluation Episode {i+1}/{eval_runs} ---")
            
            # Environment reset
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result
                
            print(f"  Initial state shape: {state.shape}")
            print(f"  Initial state range: [{state.min():.3f}, {state.max():.3f}]")
            print(f"  Initial state mean: {state.mean():.3f}")
            
            rewards = 0
            step = 0
            max_steps = 1000
            action_samples = []
            reward_samples = []
            
            while step < max_steps: 
                # Get action with debugging for first few steps
                if step < 3:
                    print(f"\n  Step {step}:")
                    print(f"    Input state range: [{state.min():.3f}, {state.max():.3f}]")
                    
                action = policy.get_action(state, eval=True)
                
                if step < 3:
                    print(f"    Action: {action[:3]}... (first 3)")
                    print(f"    Action range: [{action.min():.3f}, {action.max():.3f}]")
                
                # Environment step
                step_result = env.step(action)
                
                if len(step_result) == 5:
                    # New gym API: (obs, reward, terminated, truncated, info)
                    next_state, reward, done, truncated, info = step_result
                    done = done or truncated
                elif len(step_result) == 4:
                    # Old gym API: (obs, reward, done, info)
                    next_state, reward, done, info = step_result
                else:
                    print(f"    🚨 Unexpected step return length: {len(step_result)}")
                    break
                
                if step < 3:
                    print(f"    Reward: {reward:.3f}")
                    print(f"    Done: {done}")
                    print(f"    Next state range: [{next_state.min():.3f}, {next_state.max():.3f}]")
                
                rewards += reward
                action_samples.append(action.copy())
                reward_samples.append(reward)
                
                state = next_state
                step += 1
                
                if done:
                    break
            
            print(f"\n  Episode {i+1} Summary:")
            print(f"    Total steps: {step}")
            print(f"    Total reward: {rewards:.3f}")
            print(f"    Avg step reward: {np.mean(reward_samples):.3f}")
            print(f"    Reward std: {np.std(reward_samples):.3f}")
            print(f"    Episode ended by: {'done/truncated' if step < max_steps else 'max_steps'}")
            
            # Action analysis
            if len(action_samples) > 0:
                actions_array = np.array(action_samples)
                print(f"    Action stats - Mean: {actions_array.mean(axis=0)[:3]}...")
                print(f"    Action stats - Std: {actions_array.std(axis=0)[:3]}...")
                print(f"    Action range: [{actions_array.min():.3f}, {actions_array.max():.3f}]")
            
            reward_batch.append(rewards)
        
        if len(reward_batch) == 0:
            return 0.0, 0.0
        
        final_mean = np.mean(reward_batch)
        final_std = np.std(reward_batch)
        
        print(f"\n🎯 EVALUATION SUMMARY:")
        print(f"  Episodes: {len(reward_batch)}")
        print(f"  Mean reward: {final_mean:.3f}")
        print(f"  Std reward: {final_std:.3f}")
        print(f"  Reward range: [{min(reward_batch):.3f}, {max(reward_batch):.3f}]")
        
        return final_mean, final_std
        
    def train_once(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size = int(self.config.get('batch_size', 1024)),
            sampler = WeightedRandomSampler(
            self.dataset.priorities,
            num_samples = len(self.dataset),
            replacement = True
        ),
        collate_fn = lambda b:{
            'indices': torch.LongTensor([x[0] for x in b]),
            'state': torch.stack([x[1] for x in b]),
            'actions': torch.stack([x[2] for x in b]),
            'next_states' : torch.stack([x[3] for x in b]),
            'rewards': torch.stack([x[4] for x in b]),
            'dones': torch.stack([x[5] for x in b])
        },
        num_workers = 4
    )
        
        env = self.dataset.env
        env.action_space.seed(seed)

        batches = 0
        average10 = deque(maxlen=10)

        # Extract dataset name for project
        dataset_name = self.dataset_name.split('/')[-1] if '/' in self.dataset_name else self.dataset_name
        project_name = f"Offline SAC Exp ({dataset_name})"
        run_name = f"{self.agent_type}_seed{seed}"
        
        with wandb.init(project=project_name, group=self.agent_type, name=run_name, config=self.config):
            agent = self.create_agent()
            normalization_stats = self.dataset.get_normalization_stats()
            #print(f" Normalization Stats: {normalization_stats}")
            agent.set_normalization_stats(normalization_stats)
            wandb.watch(agent, log = "gradients", log_freq = 10)
            if self.config.get('log_video', False):
                env = gym.wrappers.RecordVideo(env, './video', episode_trigger=lambda x: True)

            eval_reward, eval_std = self.evaluate(env, agent)
            sample_batch = next(iter(self.dataloader))
            sample_states = sample_batch['state']
            sample_actions = sample_batch['actions']
            avg_rank = log_approximate_rank(agent, sample_states, sample_actions, num_samples = 10, sample_size=(64,64), delta = 0.01)
            wandb.log({"Test Reward": eval_reward, "Reward Std": eval_std, "Episode": 0, "Batches": batches, "Avg_Rank" : avg_rank}, step = batches)
            episodes = self.config.get("episodes", 100)
            for i in range(1, episodes + 1):
                agent.set_episode(i)
                if i <= self.config.get('bc_warmup_episodes', 15):
                    print(f"🔧 BC warmup episode {agent.current_episode} of {self.config.get('bc_warmup_episodes', 15)}")
                for batch_idx, batch in enumerate(self.dataloader):
                    # Extract batch data
                    states = batch['state']
                    actions = batch['actions'] 
                    rewards = batch['rewards']
                    next_states = batch['next_states']
                    dones = batch['dones']
                    
                    # Move to device
                    device = self.config.get('device', 'cuda')
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device)
                    next_states = next_states.to(device)
                    dones = dones.to(device)
                    
                    # Call agent.learn with proper format
                    policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn((states, actions, rewards, next_states, dones))
                    batches += 1

                if i % self.config.get('eval_every', 1) == 0:
                    eval_reward, eval_std = self.evaluate(env, agent)
                    sample_batch = next(iter(self.dataloader))
                    sample_states = sample_batch['state']
                    sample_actions = sample_batch['actions']
                    avg_rank = log_approximate_rank(agent, sample_states, sample_actions, num_samples=10, sample_size=(64, 64), delta=0.01)
                    wandb.log({"Test Reward": eval_reward, "Reward Std": eval_std, "Episode": i, "Batches": batches, "Avg_Rank" : avg_rank}, step=batches)

                    average10.append(eval_reward)
                    print("Episode: {} | Reward: {} | Policy Loss: {} | Batches: {} | Avg_Rank: {}".format(i, eval_reward, policy_loss, batches, avg_rank))
                
                wandb.log({
                        "Average10": np.mean(average10) if len(average10) > 0 else 0.0,
                        "Policy Loss": policy_loss,
                        "Alpha Loss": alpha_loss,
                        "Lagrange Alpha Loss": lagrange_alpha_loss,
                        "CQL1 Loss": cql1_loss,
                        "CQL2 Loss": cql2_loss,
                        "Bellman error 1": bellmann_error1,
                        "Bellman error 2": bellmann_error2,
                        "Alpha": current_alpha,
                        "Lagrange Alpha": lagrange_alpha,
                        "Batches": batches,
                        "Episode": i})

                if (i %10 == 0) and self.config.get('log_video', False):
                    mp4list = glob.glob('video/*.mp4')
                    if len(mp4list) > 1:
                        mp4 = mp4list[-2]
                        wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

                if i % self.config.get('save_every', 50) == 0:
                    # Save model checkpoint
                    os.makedirs("models", exist_ok=True)
                    save_path = f"models/{self.agent_type}_checkpoint_episode_{i}.pt"
                    agent.save(save_path)
                    print(f"Model saved at episode {i}: {save_path}")

    def train(self, seeds):
        for seed in seeds:
            print("training seed:", seed)
            self.train_once(seed)

def main():
    parser = argparse.ArgumentParser(description='Train offline RL agents with Minari datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--agent', type=str, required=True, choices=['sac', 'cqlsac', 'svrl'], help='Agent type')
    parser.add_argument('--dataset', type=str, required=True, help='Minari dataset name (e.g., halfcheetah-medium-v2)')
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated seeds or single seed')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of trials if using auto-generated seeds')
    
    args = parser.parse_args()
    
    # Parse seeds
    if ',' in args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seed_start = int(args.seeds)
        seeds = list(range(seed_start, seed_start + args.num_trials))
    
    print(f"Starting training with agent: {args.agent}")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    print(f"Seeds: {seeds}")
    
    # Create trainer and run training
    trainer = Trainer(args.dataset, args.agent, args.config)
    trainer.train(seeds)
    
    print("Training completed!")

if __name__ == "__main__":
    main()