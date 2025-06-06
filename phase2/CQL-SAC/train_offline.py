import torch
import numpy as np
import random
import wandb
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

import minari
from agent import CQLSAC  # ✅ 可替换成 UA-LQE, SVRL
from utils.utils import save
from utils.rank import log_approximate_rank
from utils.metrics import evaluate_metrics

# ---------------------------- DataLoader ----------------------------
def prep_dataloader(env_id, batch_size, seed):
    dataset = minari.load_dataset(env_id)

    obs_list, actions, rewards, next_obs_list, terminals = [], [], [], [], []
    for ep in dataset.iterate_episodes():
        obs = ep.observations[:-1]
        next_obs = ep.observations[1:]
        dones = np.logical_or(ep.terminations, ep.truncations)

        obs_list.append(obs)
        actions.append(ep.actions)
        rewards.append(ep.rewards)
        next_obs_list.append(next_obs)
        terminals.append(dones)

    obs = torch.from_numpy(np.concatenate(obs_list)).float()
    next_obs = torch.from_numpy(np.concatenate(next_obs_list)).float()

    obs_mean = obs.mean(0, keepdim=True)
    obs_std = obs.std(0, keepdim=True) + 1e-6
    obs = (obs - obs_mean) / obs_std
    next_obs = (next_obs - obs_mean) / obs_std

    actions = torch.from_numpy(np.concatenate(actions)).float()
    rewards = torch.from_numpy(np.concatenate(rewards)).float().unsqueeze(1)
    dones = torch.from_numpy(np.concatenate(terminals)).float().unsqueeze(1)

    loader = DataLoader(TensorDataset(obs, actions, rewards, next_obs, dones), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    env = dataset.recover_environment()
    return loader, env, dataset, obs, actions, obs_mean, obs_std

# ---------------------------- Evaluate ----------------------------
def evaluate(env, agent, obs_mean, obs_std, eval_runs=5):
    returns = []

    obs_mean_np = obs_mean.cpu().numpy()
    obs_std_np = obs_std.cpu().numpy()

    for _ in range(eval_runs):
        state, _ = env.reset()
        state = ((state - obs_mean_np) / obs_std_np).squeeze(0)

        episode_return = 0
        done = False

        while not done:
            action = agent.get_action(state, eval=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_return += reward

            state = ((next_state - obs_mean_np) / obs_std_np).squeeze(0)
            done = terminated or truncated

        returns.append(episode_return)

    return np.mean(returns)


# ---------------------------- Training ----------------------------
def train(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    dataloader, env, dataset, obs, actions, obs_mean, obs_std = prep_dataloader(config.env, config.batch_size, config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = dataset.observation_space.shape[0]
    action_dim = dataset.action_space.shape[0]

    agent = CQLSAC(state_dim, action_dim, config.tau, config.hidden_size, config.lr, config.temp, config.with_lagrange, config.cql_weight, config.target_gap, device)

    # ✅ Warmup BC
    warmup_loader = DataLoader(TensorDataset(obs, actions), batch_size=1024, shuffle=True, num_workers=8, pin_memory=True)
    print("[Warmup] BC Pretraining ...")
    for epoch in range(5):
        for s, a in warmup_loader:
            s, a = s.to(device), a.to(device)
            pred, _ = agent.actor_local.evaluate(s)
            loss = torch.nn.functional.mse_loss(pred, a)
            agent.actor_optimizer.zero_grad()
            loss.backward()
            agent.actor_optimizer.step()
        print(f"  [Warmup epoch {epoch+1}] done")
    print("[Warmup] Done.\n")

    batches = 0
    average10 = deque(maxlen=10)
    wandb.watch(agent, log="gradients", log_freq=10)

    # ---- Initial Eval ----
    eval_reward = evaluate(env, agent, obs_mean, obs_std)
    wandb.log({"Reward": eval_reward, "Episode": 0, "Batches": 0})


    # ---- Training Loop ----
    for ep in range(1, config.episodes + 1):
        for batch in dataloader:
            s, a, r, ns, d = [x.to(device) for x in batch]
            res = agent.learn((s, a, r, ns, d))
            (actor_loss, alpha_loss, q1_loss, q2_loss, cql1, cql2, alpha, cql_alpha_loss, cql_alpha) = res
            batches += 1

        log_data = {
            "Actor Loss": actor_loss,
            "Alpha Loss": alpha_loss,
            "Q1 Loss": q1_loss,
            "Q2 Loss": q2_loss,
            "CQL Q1": cql1,
            "CQL Q2": cql2,
            "Alpha": alpha,
            "CQL Alpha Loss": cql_alpha_loss,
            "CQL Alpha": cql_alpha,
            "Episode": ep,
            "Batches": batches,
        }

    
        if ep % config.eval_every == 0:
            # ---- Reward Eval ----
            eval_reward = evaluate(env, agent, obs_mean, obs_std)
            average10.append(eval_reward)
            log_data.update({
                "Reward" : eval_reward,
                "Average10": np.mean(average10),
            })
            print(f"[Eval] Ep {ep} | Reward: {eval_reward:.2f}")
    
            # ---- Approximate Rank ----
            if config.eval_rank:
                avg_rank = log_approximate_rank(agent, obs, actions, wandb_step=ep, num_samples=10, sample_size=(64, 64))
                log_data["Approximate Rank"] = avg_rank
                print(f"[Rank] Ep {ep} | Approx Rank: {avg_rank:.2f}")
    
            # ---- Success Rate / Distance / BC ----
            if config.eval_further:
                success_rate, distance_reduction, bc_score = evaluate_metrics(env, agent, dataset, eval_runs=10, obs_mean=obs_mean, obs_std=obs_std)
                log_data.update({
                    "Success Rate": success_rate,
                    "Goal Distance Reduction": distance_reduction,
                    "BC Score": bc_score,
                })
                print(f"[Metrics] Ep {ep} | Success Rate: {success_rate:.2f} | Dist Red: {distance_reduction:.2f}")
    
        # ✅ 统一 log 一次
        wandb.log(log_data)

        if ep % config.save_every == 0:
            save(config, "offline_agent", agent.actor_local, wandb, ep)

        agent.current_epoch += 1

# ---------------------------- Main ----------------------------
if __name__ == "__main__":
    for seed in [1101, 927, 1229, 618, 45]:
        class Config:
            def __init__(self, seed):
                self.run_name = f"CQL_SAC_hopper_seed{seed}"
                self.env = 'mujoco/hopper/medium-v0'
                self.episodes = 500
                self.seed = seed
                self.batch_size = 256
                self.hidden_size = 256
                self.lr = 1e-4
                self.tau = 2e-3
                self.eval_every = 2
                self.save_every = 100
                self.eval_further = 0
                self.eval_rank = 1
                self.temp = 1.0
                self.with_lagrange = True
                self.cql_weight = 1.0
                self.target_gap = 2.0

        config = Config(seed)
        print(f"Running Seed: {seed}")
        wandb.init(project="SAC offline (Hopper)", group='CQL-SAC offline', name=config.run_name, config=vars(config))
        train(config)
        wandb.finish()

