import torch 
import numpy as np


def evaluate_metrics(env, agent, dataset, eval_runs=10):
    success_count = 0
    total_distance_reduction = 0
    bc_score_accum = 0

    for _ in range(eval_runs):
        obs_dict, _ = env.reset()
        state = obs_dict["observation"] if "observation" in obs_dict else obs_dict

        # 初始距离
        info = {"achieved_goal": obs_dict.get("achieved_goal", None),
                "desired_goal": obs_dict.get("desired_goal", None)}
        if info["achieved_goal"] is not None and info["desired_goal"] is not None:
            initial_dist = np.linalg.norm(info["achieved_goal"] - info["desired_goal"])
        else:
            initial_dist = None

        done = False
        while not done:
            action = agent.get_action(state, eval=True)
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            state = next_obs_dict["observation"] if "observation" in next_obs_dict else next_obs_dict

            done = terminated or truncated

            # 动态更新距离（用 achieved_goal 信息）
            if "achieved_goal" in info and "desired_goal" in info:
                achieved_goal = info["achieved_goal"]
                desired_goal = info["desired_goal"]
                distance = np.linalg.norm(achieved_goal - desired_goal)

                if distance < 0.05:  # ✅ 成功条件
                    success_count += 1
                    break  # 任务成功，提前退出

                if initial_dist is not None:
                    distance_reduction = initial_dist - distance
                    total_distance_reduction += distance_reduction

            # ✅ BC score （行为克隆距离，仅示例，实际BC定义你可以改成log_prob loss）
            bc_score_accum += np.linalg.norm(action)

    success_rate = success_count / eval_runs
    avg_distance_reduction = total_distance_reduction / eval_runs
    avg_bc_score = bc_score_accum / eval_runs

    return success_rate, avg_distance_reduction, avg_bc_score

