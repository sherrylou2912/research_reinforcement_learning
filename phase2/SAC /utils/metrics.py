import numpy as np
import torch

def evaluate_metrics(env, agent, dataset, eval_runs=10, obs_mean=None, obs_std=None):
    success_count = 0
    total_distance_reduction = 0
    bc_score_accum = 0

    obs_mean_np = obs_mean.cpu().numpy() if obs_mean is not None else None
    obs_std_np = obs_std.cpu().numpy() if obs_std is not None else None

    for _ in range(eval_runs):
        obs_dict, _ = env.reset()
        state = np.concatenate([obs_dict["observation"], obs_dict["desired_goal"]])
        if obs_mean_np is not None and obs_std_np is not None:
            state = ((state - obs_mean_np) / obs_std_np).squeeze(0)

        # 初始距离
        info = {"achieved_goal": obs_dict.get("achieved_goal", None),
                "desired_goal": obs_dict.get("desired_goal", None)}
        if info["achieved_goal"] is not None and info["desired_goal"] is not None:
            initial_dist = np.linalg.norm(info["achieved_goal"] - info["desired_goal"])
        else:
            initial_dist = None

        done = False
        step_count = 0
        bc_score_episode = 0
        final_dist = initial_dist  # 初始化最终距离

        while not done:
            action = agent.get_action(state, eval=True)
            next_obs_dict, reward, terminated, truncated, info = env.step(action)

            state = np.concatenate([next_obs_dict["observation"], next_obs_dict["desired_goal"]])
            if obs_mean_np is not None and obs_std_np is not None:
                state = ((state - obs_mean_np) / obs_std_np).squeeze(0)

            done = terminated or truncated

            # 更新最新距离
            if "achieved_goal" in info and "desired_goal" in info:
                distance = np.linalg.norm(info["achieved_goal"] - info["desired_goal"])

                if distance < 0.05:  # ✅ 成功判定
                    success_count += 1
                    final_dist = 0.0  # 成功时距离记为0
                    break  # 提前退出

                final_dist = distance  # 更新到当前最近距离

            # ✅ 累加BC-like action norm
            bc_score_episode += np.linalg.norm(action)
            step_count += 1

        # ✅ episode结束后，计算距离缩减
        if initial_dist is not None:
            distance_reduction = initial_dist - final_dist
            total_distance_reduction += distance_reduction

        # ✅ 平均BC score (per step)
        if step_count > 0:
            bc_score_accum += bc_score_episode / step_count

    # ✅ 最终返回指标
    success_rate = success_count / eval_runs
    avg_distance_reduction = total_distance_reduction / eval_runs
    avg_bc_score = bc_score_accum / eval_runs

    return success_rate, avg_distance_reduction, avg_bc_score
