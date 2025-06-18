# Low-Rank Structure Modeling in Offline Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

This project investigates low-rank structure modeling and uncertainty quantification methods in offline reinforcement learning, progressively exploring and improving algorithm performance through multiple research phases.

## 📑 Project Overview

Offline reinforcement learning faces key challenges including distribution shift and value function extrapolation. This project explores the low-rank structural properties of Q-functions combined with uncertainty quantification methods to enhance the performance and stability of offline RL algorithms.

## 🔄 Research Phases

### Phase 1: Basic Algorithm Implementation and Validation
- Implemented base SAC (Soft Actor-Critic) algorithm
- Implemented CQL (Conservative Q-Learning) algorithm
- Conducted preliminary experiments on Hopper and Pendulum environments
- Established experiment logging and result visualization system

### Phase 2: Algorithm Extension and Structure Exploration
- Implemented SVRL-SAC (Structured Value-based RL) algorithm
- Extended CQL-SAC implementation
- Added matrix rank analysis tools
- Implemented basic low-rank structure modeling methods

### Phase 3: Uncertainty Quantification and Algorithm Improvements
- Implemented CQL-CVRL_SAC algorithm variant
- Added uncertainty estimation modules
- Improved training pipeline and model saving mechanism
- Implemented UALQE-SAC (Uncertainty-aware Low-rank Q-Matrix Estimation)

### Phase 4: Modular Refactoring and Extensibility Enhancement
- Refactored project structure with modular design
- Standardized data loading interfaces (D4RL and Minari dataset support)
- Unified configuration management system
- Added multiple matrix completion methods
- Enhanced uncertainty quantification modules

## 🔬 Experimental Plan

### 1. Offline RL Benchmark Evaluation

#### A. High-Dimensional Manipulation
- **Adroit Hand Tasks** (24-DoF, Complex Dynamics):
  - `pen-human-v1` (25 human demos)
  - `pen-expert-v1` (5000 expert demos)
  - `door-human-v1` (25 human demos)
  - `door-expert-v1` (5000 expert demos)
  - `hammer-human-v1` (25 human demos)
  - `hammer-expert-v1` (5000 expert demos)
  - Combined datasets: `pen-all-v0`, `door-all-v0`, `hammer-all-v0`

#### B. Locomotion and Navigation
- **AntMaze Navigation** (Sparse Rewards):
  - `antmaze-umaze-v2` (Simple U-maze)
  - `antmaze-medium-diverse-v2` (Medium maze)
  - `antmaze-large-diverse-v2` (Large maze)
- **Quadruped Locomotion**:
  - `ant-medium-v2`
  - `ant-medium-expert-v2`
  - `ant-medium-replay-v2`

#### C. Robotic Manipulation
- **Franka Kitchen** (Multi-stage Tasks):
  - `kitchen-complete-v0`
  - `kitchen-partial-v0`
  - `kitchen-mixed-v0`
- **Block Manipulation**:
  - `block-stacking-v0`
  - `block-picking-v0`
  - `block-assembly-v0`

#### D. Multi-Task Transfer Learning
- **Meta-World Tasks** (Diverse Task Structure):
  - Initial Training:
    - `MT10` (10 basic manipulation tasks)
    - Focus on rank structure learning
  - Transfer Evaluation:
    - `MT50` (50 manipulation tasks)
    - Test rank preservation and adaptation
  - Zero-shot Generalization:
    - Novel task combinations
    - Unseen object variations

#### Algorithms Comparison
- **Base Models**:
  - SAC (`phase4/agent/sac.py`)
  - CQL (`phase4/agent/cql.py`)
- **Low-rank Methods**:
  - SVRL (`phase4/agent/svrl.py`)
    - Random mask strategy
    - SoftImpute completion
    - Adaptive rank constraints
- **Uncertainty-aware Methods**:
  - UA-LQE (`phase4/agent/ua_lqe.py`)
    - Bootstrap ensemble uncertainty
    - Selective matrix completion
    - Dynamic TD-recon balancing

#### Evaluation Protocol
- **Performance Metrics**:
  - Task Success Rate
  - Average Return per Task
  - Learning Efficiency
  - Policy Coverage Analysis

- **Structure Analysis**:
  - Q-matrix Rank Evolution
  - Singular Value Distribution
  - Cross-task Knowledge Transfer
  - Value Function Decomposition

- **Uncertainty Measures**:
  - Bootstrap Variance Analysis
  - Out-of-distribution Detection
  - Uncertainty-guided Exploration
  - Value Prediction Calibration

#### Specific Experimental Settings

1. **High-Dimensional Manipulation**
   ```python
   # Using Minari for data loading
   datasets = {
       'human': minari.load_dataset('D4RL/pen-human-v1'),
       'expert': minari.load_dataset('D4RL/pen-expert-v1'),
       'combined': minari.load_dataset('D4RL/pen-all-v0')
   }
   ```
   - Focus on low-rank structure in high-dim spaces
   - Compare human vs expert demonstrations
   - Test robustness to demonstration quality

2. **Navigation and Locomotion**
   - Analyze long-horizon planning
   - Test sparse reward scenarios
   - Study value function decomposition
   - Evaluate exploration strategies

3. **Robotic Manipulation**
   - Multi-stage task completion
   - Compositional skill learning
   - Object interaction complexity
   - Task sequence handling

4. **Transfer Learning**
   - Initial training on MT10
   - Progressive transfer to MT50
   - Zero-shot generalization tests
   - Cross-task structure analysis

#### Implementation Details
```yaml
# Training Configuration
episodes_per_task: 2000
eval_episodes: 100
batch_size: 1024  # Larger for complex tasks
hidden_sizes: [400, 300]  # Deeper network

# Algorithm-specific
svrl:
  mask_ratio: 0.3
  rank_penalty: 0.01
  completion_method: "soft_impute"

ua_lqe:
  n_ensemble: 5
  uncertainty_threshold: 0.1
  lambda_schedule: "adaptive"
```

### 2. Low-Rank Structure Analysis

#### Matrix Completion Methods (`phase4/completion/`)
- **SoftImpute Implementation**
  - Varying regularization parameters
  - Convergence analysis
  - Rank estimation accuracy

#### Rank Analysis Tools (`phase4/utils/rank_utils.py`)
- Singular Value Distribution
- Effective Rank Estimation
- Temporal Difference Rank Evolution

### 3. Uncertainty Quantification Studies

#### Methods (`phase4/uncertainty/`)
- **Bootstrap Ensemble**
  - Ensemble size: 5, 10, 20
  - Prediction variance analysis
  - Uncertainty calibration

- **Value Function Uncertainty**
  - Q-value confidence bounds
  - State-action visitation uncertainty
  - Distribution shift detection

### 4. Ablation Studies

#### Component Analysis
- Impact of rank constraints
- Effect of uncertainty thresholds
- Influence of data normalization (`phase4/data/d4rl_loader.py`)

#### Hyperparameter Sensitivity
- Learning rates: [1e-4, 3e-4, 1e-3]
- Batch sizes: [256, 512, 1024]
- Network architectures: [256, 256], [400, 300]

### Implementation Details

#### Training Configuration
```yaml
# Base configuration (from phase4/config/)
episodes: 1000
batch_size: 256
learning_rate: 3e-4
gamma: 0.99
hidden_size: 256
```

#### Data Processing
- State normalization
- Reward scaling
- Automatic dataset loading via Minari

#### Logging and Monitoring
- WandB integration
- Periodic model checkpoints
- Performance metrics tracking

## 📚 References

1. SVRL: [Harnessing Structures for Value-Based Planning and Reinforcement Learning](https://arxiv.org/abs/1909.12255)
2. UA-LQE: [Uncertainty-aware Low-Rank Q-Matrix Estimation for Deep Reinforcement Learning](https://arxiv.org/pdf/2111.10103)
3. CQL: [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)

## 👥 Contributors

- [Your Name] - Project Lead

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


--- 
## Literature Review
### Harnessing Structures for Value-Based Planning and Reinforcement Learning
    proposed off-policy SVRL: random mask + matrix estimation 
    https://arxiv.org/abs/1909.12255

### Uncertainty-aware Low-Rank Q-Matrix Estimation for Deep Reinforcement Learning
    investigate the relationship between uncertainty, rank and training step 
    proposed off-policy UA-LQE: uncertainty mask + matrix estimation 
    https://arxiv.org/pdf/2111.10103

## Experimental Environments

1. Adroit Hand Tasks:
- pen-human-v0: Control a robotic hand to write with a pen
  Benchmark Performance: SAC (6.3), CQL (37.5)
- door-human-v0: Control a robotic hand to open a door
  Benchmark Performance: SAC (3.9), CQL (9.9)
- relocate-human-v0: Control a robotic hand to relocate objects
  Benchmark Performance: SAC (0.0), CQL (0.2)
- hammer-human-v0: Control a robotic hand to use a hammer
  Benchmark Performance: SAC (0.5), CQL (4.4)

2. Franka Kitchen Tasks:
- kitchen-complete-v0: Complete kitchen manipulation tasks
  Benchmark Performance: SAC (15.0), CQL (43.8)
- kitchen-partial-v0: Partial kitchen manipulation tasks
  Benchmark Performance: SAC (0.0), CQL (49.8)
- kitchen-mixed-v0: Mixed kitchen manipulation tasks
  Benchmark Performance: SAC (2.5), CQL (51.0)

3. MuJoCo Locomotion Tasks:
- hopper-medium-v0: Single-legged hopping robot
  Benchmark Performance: SAC (0.8), CQL (86.6)
- walker2d-medium-v0: Bipedal walking robot
  Benchmark Performance: SAC (0.9), CQL (74.5)
- halfcheetah-medium-v0: Cheetah-like robot
  Benchmark Performance: SAC (-4.3), CQL (44.4)

## Implementation Details

1. Network Architecture:
- Actor Network: MLP with hidden layers [256, 256]
- Critic Network: MLP with hidden layers [256, 256]
- Learning rates: 3e-4 for both actor and critic

2. Training Parameters:
- Batch size: 256
- Discount factor (gamma): 0.99
- Target network update rate (tau): 0.005
- Automatic entropy tuning with target entropy -dim(A)

3. Data Collection:
- Expert demonstrations from human operators
- Scripted policies for basic behaviors
- Mixed policy data from various training stages

4. Evaluation Metrics:
- Success rate: Percentage of successful task completions
- Average return: Mean cumulative reward per episode
- Learning efficiency: Sample efficiency and convergence speed

## References

1. Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative q-learning for offline reinforcement learning. arXiv preprint arXiv:2006.04779.

2. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning (pp. 1861-1870).

3. Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2020). D4RL: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219.






