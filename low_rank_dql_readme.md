# Low-Rank Structure in Deep Q-Learning / 深度Q学习中的低秩结构研究

[English](#english) | [中文](#chinese)

---

## English

A comprehensive research project investigating the effectiveness of low-rank matrix decomposition techniques in offline and online reinforcement learning algorithms, with a focus on Deep Q-Learning (DQL) variants.

### Overview

This project explores how low-rank structures can improve value-based planning and reinforcement learning by reducing computational complexity while maintaining or enhancing performance. We implement and compare multiple state-of-the-art algorithms that leverage low-rank approximations in Q-function estimation, with particular emphasis on offline reinforcement learning settings.

### Algorithms Implemented

#### Core Algorithms
- **SAC (Soft Actor-Critic)**: Baseline online reinforcement learning algorithm using maximum entropy framework
- **CQL-SAC**: Conservative Q-Learning with SAC, serving as our offline RL benchmark that adopts either a constraint or a penalty term that explicitly guides the policy to stay close to the given dataset
- **SVRL (Structure-aware Value-based Reinforcement Learning)**: Offline version implementing low-rank structure exploitation by leveraging Matrix Estimation (ME) techniques to exploit the underlying low-rank structure in Q functions
- **Naive-SAC**: Standard SAC implementation for comparison in offline settings

#### Planned Implementations
- **EDAC (Ensemble Diversified Actor Critic)**: Uncertainty-based offline RL with diversified Q-ensemble that takes into account the confidence of the Q-value prediction and does not require any estimation or sampling of the data distribution
- **EDUAQ (Ensemble Diversified Uncertainty-Aware Q-learning)**: Low-rank Q-matrix estimation with uncertainty quantification using N-1 Q-ensemble and 1 value reference, revealing a positive correlation between value matrix rank and value estimation uncertainty

### Key References

1. **SVRL**: "Harnessing Structures for Value-Based Planning and Reinforcement Learning" (ICLR 2020, Oral) by Yuzhe Yang, Guo Zhang, Zhi Xu, and Dina Katabi
2. **EDAC**: "Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble" (NeurIPS 2021) by Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song  
3. **EDUAQ**: "Uncertainty-aware Low-Rank Q-Matrix Estimation for Deep Reinforcement Learning" (DAI 2021) by Tong Sang, Hongyao Tang, Jianye Hao, Yan Zheng, and Zhaopeng Meng

### Current Development Status

#### Completed Work
- Implementation of SVRL-T (target-based variant), CQL-SAC, and Naive-SAC
- Testing on HalfCheetah Medium dataset from D4RL benchmark
- Performance analysis and comparative evaluation with detailed metrics tracking

#### Key Experimental Findings

**Algorithm Performance Analysis:**

Based on our experimental results on HalfCheetah Medium environment:

1. **Naive SAC Performance**: Completely fails in offline settings, maintaining near-zero reward throughout training (~0 test reward), confirming the fundamental challenge of distribution shift in offline RL.

2. **CQL-SAC Robustness**: Demonstrates superior stability and performance without entropy regularization term. Shows consistent learning progression from early stages, reaching ~5000 test reward with relatively low variance, indicating robust training dynamics.

3. **SVRL-T Superiority vs. Instability**: Outperforms CQL-SAC in final performance when entropy term is removed, achieving similar peak rewards (~5000). However, exhibits significantly higher variance in learning curves with dramatic performance jumps around 150k training steps, suggesting less stable optimization landscape.

**Low-Rank Structure Dynamics:**

Our rank analysis reveals interesting structural properties:
- **SVRL-T**: Rank evolution from ~5 to ~13-14, showing controlled rank growth during learning
- **CQL-SAC**: Starts with high rank (~17), converges to ~11-12, demonstrating rank reduction through training
- **Naive-SAC**: Maintains consistently low rank (~4-7), but with poor performance, indicating rank alone is insufficient

**Critical Insight**: SVRL-T approximates final rank values close to CQL-SAC (~13 vs ~12), suggesting effective low-rank structure utilization, but the learning path differs significantly in stability.

### Performance Metrics

The project tracks two primary metrics aligned with low-rank structure analysis:
1. **Test Reward**: Measures policy performance on evaluation episodes
2. **Average Rank**: Monitors the effective rank of Q-function approximations using singular value decomposition

### Theoretical Framework

Our approach builds on the hypothesis that if the underlying system dynamics lead to some global structures of the Q function, one should be capable of inferring the function better by leveraging such structures, specifically investigating the low-rank structure which widely exists for big data matrices.

The key insight from EDUAQ work reveals that decreasing rank of Q-matrix widely exists during learning process across a series of continuous control tasks, hypothesizing that the low-rank phenomenon indicates the common learning dynamics of Q-matrix from stochastic high dimensional space to smooth low dimensional space.

### Future Development

#### Immediate Priorities
1. **Complete Algorithm Suite**: 
   - Finish implementation of SVRL-E (evaluation-based variant)
   - Implement EDAC with ensemble-diversified actor-critic algorithm that reduces the number of required ensemble networks down to a tenth compared to the naive ensemble
   - Develop EDUAQ with uncertainty-aware reconstruction to better reduce the value estimation errors

2. **Reconstruction Method Comparison**: 
   - Target matrix reconstruction analysis (current SVRL-T approach)
   - Evaluation reconstruction assessment (planned SVRL-E)
   - Combined reconstruction approach investigation
   - Theoretical analysis of reconstruction error bounds

3. **Stability Improvements**: Address variance issues in SVRL-T learning curves through:
   - Adaptive rank regularization
   - Improved matrix completion algorithms
   - Ensemble uncertainty estimation integration

#### Extended Evaluation Plan
- **D4RL MuJoCo Tasks**: Comprehensive testing across locomotion environments (HalfCheetah, Hopper, Walker2d, Ant)
- **D4RL Adroit Tasks**: Evaluation on manipulation tasks (pen, hammer, door, relocate)
- **Multi-Quality Datasets**: Testing across expert, medium, medium-replay, and medium-expert data qualities
- **Ablation Studies**: Systematic analysis of rank constraints, reconstruction frequencies, and ensemble sizes

#### Advanced Research Directions
1. **Adaptive Rank Learning**: Dynamic adjustment based on uncertainty indicators where value estimation uncertainty can be used as the indicator of the target entries in Q-value matrix
2. **Theoretical Guarantees**: Convergence analysis for low-rank Q-function approximation in offline settings
3. **Multi-Task Transfer**: Leveraging shared low-rank structures across related control tasks
4. **Online-to-Offline Bridge**: Methods that do not require accurate estimation of the behavior policy or sampling from OOD data points
5. **Computational Efficiency**: Memory-efficient matrix operations and parallel ensemble training
6. **Safety Integration**: Uncertainty quantification for risk-aware decision making in safety-critical domains

### Installation and Usage

```bash
# Clone the repository
git clone [repository-url]
cd low-rank-dql

# Install dependencies
pip install -r requirements.txt
pip install d4rl  # For benchmark environments

# Run experiments
python train.py --algorithm svrl-t --env halfcheetah-medium-v2 --seed 42
python train.py --algorithm cql-sac --env halfcheetah-medium-v2 --seed 42

# Evaluate trained models
python evaluate.py --model_path ./results/svrl-t/model.pth --env halfcheetah-medium-v2
```

### Project Structure

```
low-rank-dql/
├── algorithms/
│   ├── sac.py              # Base SAC implementation
│   ├── cql_sac.py          # Conservative Q-Learning
│   ├── svrl.py             # Structure-aware VRL (T and E variants)
│   ├── edac.py             # Ensemble Diversified Actor-Critic
│   ├── eduaq.py            # Uncertainty-Aware Low-Rank Q-learning
│   └── naive_sac.py        # Baseline implementation
├── utils/
│   ├── matrix_estimation.py # Low-rank matrix completion utilities
│   ├── uncertainty.py       # Ensemble uncertainty quantification
│   └── rank_analysis.py     # SVD and rank monitoring tools
├── experiments/
│   ├── configs/            # Algorithm hyperparameters
│   ├── results/            # Training logs and checkpoints
│   └── plots/              # Performance visualization scripts
├── tests/
└── docs/
```

### Experimental Protocol

Our evaluation follows rigorous experimental standards:
- **5 random seeds** per algorithm-environment combination
- **1M training steps** with evaluation every 5k steps
- **10 evaluation episodes** per checkpoint
- **Standardized D4RL environments** for reproducibility
- **Statistical significance testing** using paired t-tests

---

## Chinese

一个综合性的研究项目，研究低秩矩阵分解技术在离线和在线强化学习算法中的有效性，重点关注深度Q学习（DQL）变体。

### 项目概述

本项目探索低秩结构如何通过降低计算复杂度来改进基于价值的规划和强化学习，同时保持或增强性能。我们实现并比较了多种利用Q函数估计中低秩近似的最先进算法，特别强调离线强化学习设置。

### 已实现算法

#### 核心算法
- **SAC (软行动者-批评家)**: 使用最大熵框架的基准在线强化学习算法
- **CQL-SAC**: 保守Q学习与SAC结合，作为我们的离线RL基准，采用约束或惩罚项来明确引导策略贴近给定数据集
- **SVRL (结构感知价值强化学习)**: 通过利用矩阵估计（ME）技术来开发Q函数中潜在低秩结构的离线版本实现
- **朴素SAC**: 用于离线设置比较的标准SAC实现

#### 计划实现算法
- **EDAC (集成多样化行动者批评家)**: 基于不确定性的离线RL，使用多样化Q集成，考虑Q值预测的置信度，不需要数据分布的任何估计或采样
- **EDUAQ (集成多样化不确定性感知Q学习)**: 使用N-1 Q集成和1个价值参考的低秩Q矩阵估计，揭示价值矩阵秩与价值估计不确定性之间的正相关关系

### 主要参考文献

1. **SVRL**: "Harnessing Structures for Value-Based Planning and Reinforcement Learning" (ICLR 2020, 口头报告) 作者：杨宇哲、张果、徐志、Dina Katabi
2. **EDAC**: "Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble" (NeurIPS 2021) 作者：安佳温、文胜勇、金长炫、宋贤五  
3. **EDUAQ**: "Uncertainty-aware Low-Rank Q-Matrix Estimation for Deep Reinforcement Learning" (DAI 2021) 作者：桑童、唐红尧、郝建业、郑燕、孟兆鹏

### 当前开发状态

#### 已完成工作
- 实现SVRL-T（基于目标的变体）、CQL-SAC和朴素SAC
- 在D4RL基准的HalfCheetah Medium数据集上进行测试
- 性能分析和比较评估，详细的指标跟踪

#### 关键实验发现

**算法性能分析：**

基于我们在HalfCheetah Medium环境中的实验结果：

1. **朴素SAC性能**: 在离线设置中完全失败，整个训练过程中保持接近零的奖励（~0测试奖励），证实了离线RL中分布偏移的根本挑战。

2. **CQL-SAC鲁棒性**: 在没有熵正则化项的情况下表现出卓越的稳定性和性能。从早期阶段开始显示一致的学习进展，达到~5000测试奖励，方差相对较低，表明训练动态稳健。

3. **SVRL-T优越性与不稳定性**: 当移除熵项时，在最终性能上优于CQL-SAC，达到相似的峰值奖励（~5000）。然而，学习曲线表现出显著更高的方差，在150k训练步骤左右出现戏剧性的性能跳跃，表明优化景观稳定性较差。

**低秩结构动态：**

我们的秩分析揭示了有趣的结构性质：
- **SVRL-T**: 秩从~5演化到~13-14，显示学习过程中受控的秩增长
- **CQL-SAC**: 从高秩（~17）开始，收敛到~11-12，展示通过训练的秩降低
- **朴素SAC**: 保持一致的低秩（~4-7），但性能差，表明仅有秩是不够的

**关键洞察**: SVRL-T的最终秩值接近CQL-SAC（~13 vs ~12），表明有效的低秩结构利用，但学习路径在稳定性上差异显著。

### 性能指标

项目跟踪与低秩结构分析对齐的两个主要指标：
1. **测试奖励**: 测量评估片段上的策略性能
2. **平均秩**: 使用奇异值分解监控Q函数近似的有效秩

### 理论框架

我们的方法建立在这样的假设上：如果底层系统动力学导致Q函数的某些全局结构，则应该能够通过利用这些结构来更好地推断函数，特别是研究在大数据矩阵中广泛存在的低秩结构。

EDUAQ工作的关键洞察揭示了在一系列连续控制任务的学习过程中广泛存在Q矩阵秩的降低，假设低秩现象表明了Q矩阵从随机高维空间到平滑低维空间的共同学习动力学。

### 未来发展

#### 即时优先项
1. **完整算法套件**：
   - 完成SVRL-E（基于评估的变体）实现
   - 实现将所需集成网络数量减少到朴素集成的十分之一的集成多样化行动者批评家算法EDAC
   - 开发具有不确定性感知重构以更好地减少价值估计误差的EDUAQ

2. **重构方法比较**：
   - 目标矩阵重构分析（当前SVRL-T方法）
   - 评估重构评估（计划中的SVRL-E）
   - 组合重构方法调研
   - 重构误差界限的理论分析

3. **稳定性改进**: 通过以下方法解决SVRL-T学习曲线中的方差问题：
   - 自适应秩正则化
   - 改进的矩阵完成算法
   - 集成不确定性估计集成

#### 扩展评估计划
- **D4RL MuJoCo任务**: 在运动环境中进行全面测试（HalfCheetah, Hopper, Walker2d, Ant）
- **D4RL Adroit任务**: 在操作任务中进行评估（pen, hammer, door, relocate）
- **多质量数据集**: 在专家、中等、中等回放和中等专家数据质量中进行测试
- **消融研究**: 对秩约束、重构频率和集成大小进行系统分析

#### 高级研究方向
1. **自适应秩学习**: 基于价值估计不确定性可以用作Q值矩阵中目标条目指标的不确定性指标进行动态调整
2. **理论保证**: 离线设置中低秩Q函数近似的收敛分析
3. **多任务迁移**: 在相关控制任务中利用共享低秩结构
4. **在线到离线桥接**: 不需要准确估计行为策略或从OOD数据点采样的方法
5. **计算效率**: 内存高效的矩阵操作和并行集成训练
6. **安全集成**: 在安全关键领域中用于风险感知决策制定的不确定性量化

### 安装和使用

```bash
# 克隆仓库
git clone [repository-url]
cd low-rank-dql

# 安装依赖项
pip install -r requirements.txt
pip install d4rl  # 用于基准环境

# 运行实验
python train.py --algorithm svrl-t --env halfcheetah-medium-v2 --seed 42
python train.py --algorithm cql-sac --env halfcheetah-medium-v2 --seed 42

# 评估训练模型
python evaluate.py --model_path ./results/svrl-t/model.pth --env halfcheetah-medium-v2
```

### 项目结构

```
low-rank-dql/
├── algorithms/
│   ├── sac.py              # 基础SAC实现
│   ├── cql_sac.py          # 保守Q学习
│   ├── svrl.py             # 结构感知VRL（T和E变体）
│   ├── edac.py             # 集成多样化行动者批评家
│   ├── eduaq.py            # 不确定性感知低秩Q学习
│   └── naive_sac.py        # 基准实现
├── utils/
│   ├── matrix_estimation.py # 低秩矩阵完成工具
│   ├── uncertainty.py       # 集成不确定性量化
│   └── rank_analysis.py     # SVD和秩监控工具
├── experiments/
│   ├── configs/            # 算法超参数
│   ├── results/            # 训练日志和检查点
│   └── plots/              # 性能可视化脚本
├── tests/
└── docs/
```

### 实验协议

我们的评估遵循严格的实验标准：
- **每个算法-环境组合5个随机种子**
- **100万训练步骤**，每5k步骤评估一次
- **每个检查点10个评估片段**
- **标准化D4RL环境**以确保可重复性
- **统计显著性测试**使用配对t检验

### 贡献指南

本项目正在积极开发中。欢迎贡献，特别是在：
- 算法实现
- 实验设计
- 性能优化
- 文档改进

### 许可证

[许可证信息待添加]

---

*本项目旨在推进对强化学习中低秩结构的理解及其在复杂控制任务中的实际应用。*

*This project aims to advance the understanding of low-rank structures in reinforcement learning and their practical applications in complex control tasks.*