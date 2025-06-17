import os
import gym
import d4rl
import torch
import yaml
import argparse
from typing import Dict, Any

from agent.sac import SAC
from agent.cql import CQL
from agent.svrl import SVRL
from data.d4rl_loader import D4RLDataset
from utils.logger import Logger
from utils.eval_metrics import evaluate_policy

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_agent(agent_type: str, env, config: Dict[str, Any]):
    """Create agent based on type."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    if agent_type == 'sac':
        return SAC(state_size, action_size, config)
    elif agent_type == 'cql':
        return CQL(state_size, action_size, config)
    elif agent_type == 'svrl':
        return SVRL(state_size, action_size, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--agent', type=str, required=True, choices=['sac', 'cql', 'svrl'])
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    
    # Create environment
    env = gym.make(config['env'])
    eval_env = gym.make(config['env'])
    
    # Create agent
    agent = create_agent(args.agent, env, config)
    
    # Create dataset
    dataset = D4RLDataset(
        config['env'],
        normalize_states=True,
        normalize_rewards=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # Create logger
    logger = Logger(
        config['project_name'],
        config,
        use_wandb=config['use_wandb']
    )
    
    # Training loop
    total_steps = 0
    for episode in range(1, config['episodes'] + 1):
        for batch_idx, batch in enumerate(dataloader):
            # Update agent
            metrics = agent.update(batch)
            total_steps += 1
            
            # Log metrics
            if total_steps % config['log_every'] == 0:
                logger.log_metrics(metrics, step=total_steps)
            
            # Evaluate
            if total_steps % config['eval_every'] == 0:
                mean_return, std_return = evaluate_policy(
                    eval_env,
                    agent,
                    n_episodes=config['eval_episodes']
                )
                logger.log_metrics({
                    'eval/mean_return': mean_return,
                    'eval/std_return': std_return
                }, step=total_steps)
                
                print(f"Episode {episode}, Step {total_steps}: Mean Return = {mean_return:.2f} ± {std_return:.2f}")
            
            # Save model
            if total_steps % config['save_every'] == 0:
                logger.log_model(agent, f"model_step_{total_steps}")
    
    # Final evaluation
    mean_return, std_return = evaluate_policy(
        eval_env,
        agent,
        n_episodes=config['eval_episodes']
    )
    logger.log_metrics({
        'eval/final_mean_return': mean_return,
        'eval/final_std_return': std_return
    })
    
    print(f"Training finished: Final Return = {mean_return:.2f} ± {std_return:.2f}")
    
    # Save final model
    logger.log_model(agent, "model_final")
    logger.finish()

if __name__ == "__main__":
    main()
