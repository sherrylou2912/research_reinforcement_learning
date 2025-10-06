import gym
import d4rl
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE
import numpy as np
from collections import deque
import torch
import wandb
import argparse
import glob
from utils import save, collect_random
import random
from tqdm import tqdm 
import requests

from agent import EDACSAC
from torch.utils.data import DataLoader, TensorDataset
from rank import log_approximate_rank

def d4rl_score(task, rew_mean):
    score = (rew_mean - REF_MIN_SCORE[task]) / (REF_MAX_SCORE[task] - REF_MIN_SCORE[task]) * 100
    return score 


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="EDAC_SAC", help="Run name prefix, default: CQL")
    parser.add_argument("--env", type=str, default="halfcheetah-medium-v2", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes, default: 100")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    parser.add_argument("--seed", type=int, default=1101, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eta", type = float, default=1.0, help="")
    parser.add_argument("--n_q_networks", type = int, default= 10, help = "")
    
    args = parser.parse_args()
    return args

def prep_dataloader(env_id="halfcheetah-medium-v2", batch_size=256, seed=1):
    env = gym.make(env_id)
    dataset = env.get_dataset()
    tensors = {}
    for k, v in dataset.items():
        if k in ["actions", "observations", "next_observations", "rewards", "terminals"]:
            if  k != "terminals":
                tensors[k] = torch.from_numpy(v).float()
            else:
                tensors[k] = torch.from_numpy(v).long()

    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"][:, None],
                               tensors["next_observations"],
                               tensors["terminals"][:, None])
    dataloader  = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
    
    if "halfcheetah" in env_id:
        eval_env = gym.make("HalfCheetah-v2")
    eval_env.seed(seed)
    return dataloader, eval_env

def evaluate(env, policy, eval_runs=5): 
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()

        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)

            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch), np.std(reward_batch)

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataloader, env = prep_dataloader(env_id=config.env, batch_size=config.batch_size, seed=config.seed)
    env.action_space.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batches = 0
    average10 = deque(maxlen=10)

    # 初始化变量避免未定义错误
    latest_eval_reward = 0.0
    latest_reward_std = 0.0
    latest_score = 0.0
    latest_score_std = 0.0
    latest_policy_loss = 0.0
    
    with wandb.init(project="Offline SAC Exp (halfcheetah-medium-v2)", group = "EDAC-SAC", name=config.run_name, 
                    mode = "offline", config=config):
        
        agent = EDACSAC(state_size=env.observation_space.shape[0],
                        action_size=env.action_space.shape[0],
                        tau=config.tau,
                        hidden_size=config.hidden_size,
                        learning_rate=config.learning_rate,
                        n_q_networks=config.n_q_networks,
                        eta= config.eta,
                        device=device)

        wandb.watch(agent, log="gradients", log_freq=10)
        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

        # 初始评估
        latest_eval_reward, latest_reward_std = evaluate(env, agent)
        latest_score = latest_reward_std * 100 / (REF_MAX_SCORE[config.env] - REF_MIN_SCORE[config.env])
        
        sample_states, sample_actions, _, _, _ = next(iter(dataloader))
        avg_rank = log_approximate_rank(agent, sample_states, sample_actions, num_samples=10, sample_size=(64, 64), delta=0.01)
        wandb.log({"Test Reward": latest_eval_reward, "Reward Std": latest_reward_std, 
                   "Test Score": latest_score, "Score Std": latest_score_std,
                   "Episode": 0, "Batches": batches, "Avg_Rank" : avg_rank}, step=batches)
        
        for i in range(1, config.episodes+1):
            pbar = tqdm(enumerate(dataloader), total = len(dataloader),
                        desc = f"Episode {i}/{config.episodes}", unit = "batch", leave = False)

            for batch_idx, experience in pbar:
                states, actions, rewards, next_states, dones = experience
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn((states, actions, rewards, next_states, dones))
                latest_policy_loss = policy_loss  # 更新最新的policy loss
                batches += 1

            if i % config.eval_every == 0:
                latest_eval_reward, latest_reward_std = evaluate(env, agent)
                latest_score = d4rl_score(config.env, latest_eval_reward)
                # 修复：score_std应该基于reward的标准差计算，不是直接用d4rl_score转换std
                latest_score_std = latest_reward_std * 100 / (REF_MAX_SCORE[config.env] - REF_MIN_SCORE[config.env])
                
                sample_states, sample_actions, _, _, _ = next(iter(dataloader))
                avg_rank = log_approximate_rank(agent, sample_states, sample_actions, num_samples=10, sample_size=(64, 64), delta=0.01)
                wandb.log({"Test Reward": latest_eval_reward, "Reward Std": latest_reward_std,
                           "Score": latest_score, "Score_std": latest_score_std, 
                           "Episode": i, "Batches": batches, "Avg_Rank" : avg_rank}, step=batches)

                average10.append(latest_eval_reward)
                print("Episode: {} | Reward: {:.2f} | Reward Std: {:.2f} | Policy Loss: {:.4f} | Batches: {}".format(
                    i, latest_eval_reward, latest_reward_std, latest_policy_loss, batches))
            
            wandb.log({
                       "Average10": np.mean(average10),
                       "Policy Loss": latest_policy_loss,  # 使用latest版本
                       "Alpha Loss": alpha_loss,
                       "Bellman error 1": bellmann_error1,
                       "Bellman error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Batches": batches,
                       "Episode": i})

            if (i %10 == 0) and config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            if i % config.save_every == 0:
                save(config, save_name="EDAC", model=agent.actor_local, wandb=wandb, ep=0)

            if i % 50 == 0:
                headers = {"Authorization" : "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjU1MDczOCwidXVpZCI6IjJkMTY0NzU1LTFjZjUtNDgwMi04YTNkLWE4ZTkyMGYwNGQ3YiIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.qP-FY_mf7kx38tRnSDZTBC8_IMMGkH9baLTJ1QYDUL1_ytTKkPGGKjCaqp0vh5EO5Q8oybxEFDkPbRSmXgEP7Q"}
                resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                     json={
                         "title": "HalfCheetah Exp (Medium v2)",  # 修正拼写错误
                         "name": "EDAC method",  # 修正方法名
                         "content": "Episode: {} | Score: {:.2f} | Score Std: {:.2f} | Policy Loss: {:.4f}".format(
                             i, latest_score, latest_score_std, latest_policy_loss)
                     }, headers = headers)
                print(resp.content.decode())

if __name__ == "__main__":
    base_config = get_config()
    seeds = [618, 45]
    for seed in seeds:
        config = argparse.Namespace(**vars(base_config))
        config.seed = seed
        config.run_name = f"{base_config.run_name}_seed{seed}"
        train(config)
