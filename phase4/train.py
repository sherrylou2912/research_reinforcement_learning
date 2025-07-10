import os
import gymnasium as gym
# import d4rl  # 移除d4rl
import torch
import yaml
import argparse
from typing import Dict, Any
import numpy as np
import time # Added for timing

from agent.sac import SAC
from agent.cql import CQL
from agent.svrl import SVRL
from data.minari_loader import create_minari_dataloader  # 用minari数据加载
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

def train_once(args, config, seed):
    """运行单次完整训练"""
    print(f"\nInitializing training with seed {seed}...")
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建数据加载器
    print("Loading dataset...")
    data_loader, normalization_stats = create_minari_dataloader(
        dataset_name=args.dataset,
        batch_size=config['batch_size'],
        normalize_states=args.normalize_states,
        normalize_rewards=args.normalize_rewards,
        device=config['device']
    )
    print(f"Dataset loaded with batch size {config['batch_size']}")
    
    # 创建环境和智能体
    print("Creating environment...")
    env = gym.make(config['env'])
    eval_env = gym.make(config['env'])
    
    # 获取维度
    sample_batch = next(iter(data_loader))
    state_size = sample_batch['observations'].shape[1]
    action_size = sample_batch['actions'].shape[1]
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # 创建智能体
    print(f"Creating {args.agent} agent...")
    if args.agent == 'sac':
        agent = SAC(state_size, action_size, config)
    elif args.agent == 'cql':
        agent = CQL(state_size, action_size, config)
    else:
        agent = SVRL(state_size, action_size, config)
    
    if args.normalize_states or args.normalize_rewards:
        agent.set_normalization_stats(normalization_stats)
        print("Normalization stats set")
    
    # 创建logger
    logger_config = {
        'algorithm': args.agent,
        'env_name': config['env'],
        'seed': seed,
        'batch_size': config['batch_size'],
        'learning_rate': config['learning_rate'],
        'normalize_states': args.normalize_states,
        'normalize_rewards': args.normalize_rewards,
        'group': config.get('group', 'default'),
        'name': f"{config.get('name', 'default')}_seed{seed}"
    }
    
    logger = Logger(
        project_name=config['project_name'],
        config=logger_config,
        output_dir=os.path.join("logs", f"{args.agent}_{seed}"),
        use_wandb=config['use_wandb']
    )
    print("Logger initialized")
    
    # 设置CUDA性能优化
    if config['device'] == 'cuda':
        print("\nOptimizing CUDA performance...")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("CUDA optimization settings applied")
    
    # 训练循环
    total_steps = 0
    best_return = float('-inf')
    episode_returns = []
    
    print("\nStarting training loop...")
    print(f"Total episodes: {config['episodes']}")
    print(f"Eval frequency: every {config['eval_every']} episodes")
    print(f"Log frequency: every {config['log_every']} steps")
    
    # 预计算总步数
    steps_per_epoch = len(data_loader)
    total_expected_steps = steps_per_epoch * config['episodes']
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total expected steps: {total_expected_steps}")
    
    start_time = time.time()
    last_log_time = start_time
    
    for episode in range(1, config['episodes'] + 1):
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(data_loader):
            # 将batch转到目标设备
            for k in batch:
                batch[k] = batch[k].to(config['device'])
            
            # 更新智能体
            metrics = agent.update(batch)
            total_steps += 1
            
            # 记录日志
            if total_steps % config['log_every'] == 0:
                current_time = time.time()
                steps_per_second = config['log_every'] / (current_time - last_log_time)
                last_log_time = current_time
                
                print(f"\rEpisode {episode}/{config['episodes']} "
                      f"[{batch_idx+1}/{len(data_loader)}] "
                      f"Steps: {total_steps}/{total_expected_steps} "
                      f"({steps_per_second:.1f} steps/s) "
                      f"Loss: {metrics['critic1_loss']:.3f}", end="")
                
                logger.log_metrics(metrics, total_steps)
        
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpisode {episode} completed in {epoch_time:.2f}s")
        
        # 评估
        if episode % config['eval_every'] == 0:
            print(f"\nEvaluating at episode {episode}...")
            mean_return, std_return = evaluate_policy(
                eval_env,
                agent,
                config['eval_episodes'],
                deterministic=True
            )
            episode_returns.append(mean_return)
            
            print(f"Evaluation results - Mean return: {mean_return:.2f} ± {std_return:.2f}")
            
            logger.log_metrics({
                'eval/mean_return': mean_return,
                'eval/std_return': std_return,
                'eval/episode': episode
            }, total_steps)
            
            # 保存最佳模型
            if mean_return > best_return:
                best_return = mean_return
                if config.get('save_best', True):
                    save_path = f"models/{args.agent}_{config['env']}_seed{seed}_best.pt"
                    agent.save(save_path)
                    print(f"New best model saved to {save_path}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s")
    print(f"Best return achieved: {best_return:.2f}")
    
    return best_return, episode_returns

def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--agent', type=str, required=True, choices=['sac', 'cql', 'svrl'])
    parser.add_argument('--dataset', type=str, required=True, help='Minari dataset name')
    parser.add_argument('--normalize_states', action='store_true', help='Whether to normalize states')
    parser.add_argument('--normalize_rewards', action='store_true', help='Whether to normalize rewards')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of complete training runs')
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated list of seeds')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 解析种子
    if ',' in args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
    else:
        seed_start = int(args.seeds)
        seeds = list(range(seed_start, seed_start + args.num_trials))
    
    # 运行多次训练
    trial_returns = []
    all_episode_returns = []
    
    print(f"\n=== Starting {len(seeds)} complete training runs ===")
    for i, seed in enumerate(seeds):
        print(f"\nTrial {i+1}/{len(seeds)} (seed={seed})")
        best_return, episode_returns = train_once(args, config, seed)
        trial_returns.append(best_return)
        all_episode_returns.append(episode_returns)
    
    # 计算并打印最终结果
    trial_mean = np.mean(trial_returns)
    trial_std = np.std(trial_returns)
    print("\n=== Final Results ===")
    print(f"Across {len(seeds)} complete trials:")
    print(f"Mean best return: {trial_mean:.2f} ± {trial_std:.2f}")
    
    # 计算学习曲线的统计信息
    episode_returns_array = np.array(all_episode_returns)
    mean_curve = np.mean(episode_returns_array, axis=0)
    std_curve = np.std(episode_returns_array, axis=0)
    print("\nLearning curve statistics:")
    print(f"Final performance: {mean_curve[-1]:.2f} ± {std_curve[-1]:.2f}")
    
    # 保存结果
    results = {
        'trial_returns': trial_returns,
        'trial_mean': trial_mean,
        'trial_std': trial_std,
        'learning_curves': all_episode_returns,
        'curve_mean': mean_curve.tolist(),
        'curve_std': std_curve.tolist(),
        'config': config,
        'args': vars(args),
        'seeds': seeds
    }
    
    import json
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results_{args.agent}_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
