import gym
import d4rl
import numpy as np
from collections import deque
import torch
import wandb
import argparse
import glob
from utils import save, collect_random
import random
import yaml
from tqdm import tqdm

from agent import CQLSAC
from data.d4rl_loader import D4rlDataset
from rank_utils import log_approximate_rank

class Trainer:
    def __init__(self, env_id, agent_type, config_path) -> None:
        self.env_id = env_id 
        self.config = self.load_config(config_path)
        self.dataset = D4rlDataset(env_id = self.env_id,
                                   batch_size=self.config.get('batch_size',256),
                                   seed = self.config.get('seed', None),
                                   eval_env_id = self.config.get('eval_env_id', None))
        
        self.agent_type = agent_type
        self.dataloader, self.env = self.dataset.prep_dataloader()
        self.env.action_space.seed(self.config.get('seed', 1229))
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.device = self.config.get('device', 'cuda')
        

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config
    
    def create_agent(self):
        if self.agent_type == 'cqlsac':
            return CQLSAC(self.state_dim, self.action_dim, self.config)
        
    def evaluate(self, policy, eval_runs=5): 
        """
        Makes an evaluation run with the current policy
        """
        reward_batch = []
        for i in range(eval_runs):
            state = self.env.reset()

            rewards = 0
            while True:
                action = policy.get_action(state, eval=True)

                state, reward, done, _ = self.env.step(action)
                rewards += reward
                if done:
                    break
            reward_batch.append(rewards)
        return np.mean(reward_batch) if reward_batch else 0.0
    
    def train_once(self, seed):

        #seet up seed 
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        batches = 0 
        average10 = deque(maxlen=10)

        project_name = f"Offline SAC Exp ({self.env_id})"
        run_name = f"{self.agent_type}_seed{seed}"
        
        with wandb.init(project = project_name, group = self.agent_type, name = run_name, config=self.config):
            agent = self.create_agent()
            wandb.watch(agent, log="gradients", log_freq=10)
            eval_reward = self.evaluate(agent)
            sample_states, sample_actions, _, _, _ = next(iter(self.dataloader))
            avg_rank = log_approximate_rank(agent, sample_states, sample_actions, num_samples=10, sample_size=(64, 64), delta=0.01)
            wandb.log({"Test Reward": eval_reward, "Episode": 0, "Batches": batches, "Avg_Rank" : avg_rank}, step=batches)
            for i in range(1, self.config.get('episodes', 200) + 1):
                print('batch')
                for batch_idx, experience in enumerate(self.dataloader):
                    states, actions, rewards, next_states, dones = experience
                    states = states.to(self.device)
                    actions = actions.to(self.device)
                    rewards = rewards.to(self.device)
                    next_states = next_states.to(self.device)
                    dones = dones.to(self.device)
                    policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn((states, actions, rewards, next_states, dones))
                    batches += 1

                if i % self.config.get('eval_every', 1) == 0:
                    eval_reward = self.evaluate(agent)
                    sample_states, sample_actions, _, _, _ = next(iter(self.dataloader))
                    avg_rank = log_approximate_rank(agent, sample_states, sample_actions, num_samples=10, sample_size=(64, 64), delta=0.01)
                    wandb.log({"Test Reward": eval_reward, "Episode": i, "Batches": batches, "Avg_Rank" : avg_rank}, step=batches)

                    average10.append(eval_reward)
                    print("Episode: {} | Reward: {} | Policy Loss: {} | Batches: {}".format(i, eval_reward, policy_loss, batches,))
                
                wandb.log({
                        "Average10": np.mean(average10),
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
                
                if i % self.config.get('save_every' , 100) == 0:
                    save(self.config, save_name="CQL", model=agent.actor_local, wandb=wandb, ep=0)

    def train(self, seeds):
        for seed in seeds:
            print('training seed: ', seed)
            self.train_once(seed)

    
def main():
    parser = argparse.ArgumentParser(description='Train offline RL agents with D4RL datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--agent', type=str, required=True, choices=['sac', 'cqlsac', 'svrl'], help='Agent type')
    parser.add_argument('--env_id', type=str, required=True, help='D4RL dataset name (e.g., halfcheetah-medium-v2)')
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
    print(f"Dataset: {args.env_id}")
    print(f"Config: {args.config}")
    print(f"Seeds: {seeds}")
    
    # Create trainer and run training
    trainer = Trainer(args.env_id, args.agent, args.config)
    trainer.train(seeds)
    
    print("Training completed!")

if __name__ == "__main__":
    main()