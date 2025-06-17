# 📦 离线强化学习中的低秩结构建模

## To Do 
- 完善 CQL-SAC code 复现论文benchmark
- 重新实验SVRL-SAC
- 实现UA-LQE-SAC

--- 
## Literature Review
### Harnessing Structures for Value-Based Planning and Reinforcement Learning
    proposed off-policy SVRL: random mask + matrix estimation 
    https://arxiv.org/abs/1909.12255

### Uncertainty-aware Low-Rank Q-Matrix Estimation for Deep Reinforcement Learning
    investigate the relationship between uncertainty, rank and training step 
    proposed off-policy UA-LQE: uncertainty mask + matrix estimation 
    https://arxiv.org/pdf/2111.10103



---

## 🧠 算法介绍

### 🧩 1. SVRL-SAC（已实现）

- 随机掩码 Q 矩阵；
- 使用 SoftImpute 等方法进行低秩矩阵补全；
- Critic 损失结合原始 TD 误差与结构重建误差。

---

### 🔍 2. UA-LQE-SAC（开发中）

- 使用 Bootstrap 或 Count-based 的不确定性来确定哪些 Q 值可信；
- 只对可信区域进行低秩补全；
- 动态融合原始 TD 与重构目标：
  \[
  Q_{\text{target}} = (1 - \lambda_t) Q_{\text{TD}} + \lambda_t Q_{\text{recon}}
  \]

---

## 🧪 实验结果

我们使用 [D4RL](https://github.com/rail-berkeley/d4rl) 提供的 **HalfCheetah-medium-v2** 数据集进行实验。

### 当前结果（SVRL）

| 方法         | 最终回报 | Q 矩阵秩 ↓ | 策略稳定性       |
|--------------|----------|------------|------------------|
| Naive SAC    | 2900     | 高         | 波动明显          |
| CQL-SAC      | **3700** | 中等       | 稳定              |
| SVRL-SAC     | 3600     | **低**     | 稳定              |

---





