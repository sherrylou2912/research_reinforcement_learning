# Configuration System Documentation

This directory contains hierarchical configuration files for the Offline Reinforcement Learning framework. The configuration system uses inheritance to minimize duplication and maintain consistency across different algorithms.

## Configuration Hierarchy

```
base.yaml           # Common settings for all algorithms
├── sac.yaml        # SAC-specific extensions
├── cql.yaml        # CQL extends SAC
│   └── svrl.yaml   # SVRL extends CQL
└── ua_lqe.yaml     # UA-LQE extends base
```

## Usage

### Command Line Usage
```bash
# Use specific algorithm config
python train.py --config config/sac.yaml --agent sac --dataset "..."
python train.py --config config/sac.yaml --agent sac --dataset "mujoco/halfcheetah/medium-v0"

```

### Configuration File Structure
Each algorithm inherits from its parent using the `defaults` key:
```yaml
defaults:
  - base  # or sac, cql, etc.
```

## Configuration Files

### `base.yaml` - Foundation Configuration
Contains common parameters shared across all algorithms:

#### Environment Settings
- **`env`**: Gymnasium environment name (e.g., "HalfCheetah-v4")
- **`seed`**: Random seed for reproducible experiments

#### Network Architecture
- **`hidden_size`**: Hidden layer dimensions for neural networks (typically 256)
- **`activation`**: Activation function ("relu", "tanh", "elu", "gelu")

#### Training Parameters
- **`episodes`**: Total training episodes
- **`batch_size`**: Mini-batch size for gradient updates
- **`learning_rate`**: Optimizer learning rate
- **`gamma`**: Discount factor for future rewards [0,1]
- **`tau`**: Target network soft update coefficient [0,1]

#### Hardware & Logging
- **`device`**: Compute device ("cuda" or "cpu")
- **`use_wandb`**: Enable Weights & Biases logging
- **`project_name`**: W&B project identifier

### `sac.yaml` - Soft Actor-Critic Configuration
Extends base configuration with SAC-specific parameters:

#### Core SAC Parameters
- **`alpha`**: Temperature parameter for entropy regularization
  - Higher values → more exploration
  - Lower values → more exploitation
- **`target_entropy`**: Target entropy for automatic temperature tuning
  - `null` sets to `-dim(action_space)`
- **`auto_tune_alpha`**: Enable automatic temperature adjustment

#### Advanced SAC Settings
- **`critic_lr`**: Critic network learning rate (can differ from actor)
- **`actor_lr`**: Actor network learning rate
- **`alpha_lr`**: Temperature parameter learning rate
- **`reward_scale`**: Reward scaling factor
- **`update_after`**: Start updates after N environment steps
- **`update_every`**: Update frequency (every N steps)

### `cql.yaml` - Conservative Q-Learning Configuration
Extends SAC for offline RL with conservative Q-value estimation:

#### Core CQL Parameters
- **`cql_alpha`**: CQL regularization coefficient
  - Higher values → more conservative Q-values
  - Typical range: [0.1, 10.0]
- **`cql_tau`**: Temperature for Boltzmann sampling in CQL loss
- **`with_lagrange`**: Enable automatic CQL coefficient tuning
- **`target_action_gap`**: Target conservative gap for Lagrange method
- **`cql_min_q_weight`**: Weight for conservative Q-loss component

#### Advanced CQL Settings
- **`n_actions_cql`**: Number of sampled actions for CQL loss computation
- **`beta`**: Behavior cloning regularization weight
- **`temp`**: Temperature for CQL action sampling
- **`max_q_backup`**: Use maximum Q-backup instead of soft backup
- **`deterministic_backup`**: Use deterministic policy for Q-backup

### `svrl.yaml` - Structured Value-based Representation Learning
Combines CQL with low-rank matrix factorization:

#### Core SVRL Parameters
- **`rank`**: Rank for low-rank matrix factorization of Q-functions
  - Lower rank → more structured representation
  - Typical range: [5, 50]
- **`svrl_weight`**: Weight for SVRL regularization loss
- **`n_action_sample`**: Actions sampled for structured loss computation
- **`mask_prob`**: Probability for masking in structured learning
- **`lambda_struct`**: Regularization weight for structure preservation

#### Matrix Completion Settings
- **`soft_impute_iters`**: Iterations for soft imputation algorithm
- **`soft_impute_threshold`**: Convergence threshold for matrix completion
- **`use_soft_impute`**: Enable soft imputation for handling missing values

#### Advanced SVRL Settings
- **`orthogonal_init`**: Initialize U,V matrices with orthogonal initialization
- **`grad_clip_norm`**: Gradient clipping for training stability
- **`struct_update_freq`**: Update structured components every N steps
- **`use_target_struct`**: Use target networks for structured components

### `ua_lqe.yaml` - Uncertainty-Aware Low-rank Q Ensemble
Advanced ensemble method for uncertainty estimation:

#### Core UA-LQE Parameters
- **`n_critics`**: Number of Q-networks in ensemble
  - More critics → better uncertainty estimation
  - Typical range: [3, 10]
- **`rank`**: Rank for low-rank Q-function approximation
- **`uncertainty_weight`**: Weight for uncertainty regularization
- **`ensemble_method`**: Uncertainty quantification method
  - "variance": Use prediction variance
  - "disagreement": Use ensemble disagreement

#### Low-rank Settings
- **`use_low_rank`**: Enable low-rank Q-function approximation
- **`rank_reg_weight`**: Regularization weight for rank constraint
- **`nuclear_norm_reg`**: Nuclear norm regularization strength

#### Ensemble Training
- **`bootstrap_ratio`**: Bootstrap sampling ratio for each critic
- **`diversity_loss_weight`**: Weight for promoting ensemble diversity
- **`update_ensemble_freq`**: Update frequency for all ensemble members

#### Advanced UA-LQE Settings
- **`use_target_ensemble`**: Use target networks for ensemble
- **`uncertainty_threshold`**: Threshold for high uncertainty detection
- **`conservative_weight`**: Conservative weighting for uncertain regions
- **`dropout_rate`**: Dropout rate for additional uncertainty

### `experiment.yaml` - Quick Testing Configuration
Simplified configuration for rapid experimentation with sensible defaults.

## Parameter Guidelines

### Choosing Hyperparameters

#### Learning Rates
- **Conservative**: 1e-4 (stable, slow convergence)
- **Standard**: 3e-4 (good balance)
- **Aggressive**: 1e-3 (fast, potentially unstable)

#### Batch Sizes
- **Small**: 64-128 (less memory, more gradient noise)
- **Medium**: 256-512 (balanced)
- **Large**: 1024+ (stable gradients, more memory)

#### Network Architecture
- **Hidden Size**: 256 (standard), 512 (complex tasks), 128 (simple tasks)
- **Activation**: "relu" (default), "tanh" (bounded), "elu" (smooth)

#### Algorithm-Specific Guidelines

**SAC**:
- `alpha`: 0.2 (standard), 0.1 (less exploration), 0.5 (more exploration)
- `tau`: 0.005 (standard), 0.001 (slower target updates)

**CQL**:
- `cql_alpha`: 1.0 (balanced), 5.0 (very conservative), 0.1 (less conservative)
- Use `with_lagrange: true` for automatic tuning

**SVRL**:
- `rank`: 10 (standard), 5 (more structured), 20 (less structured)
- `svrl_weight`: 1.0 (balanced), 0.1 (less structure), 2.0 (more structure)

**UA-LQE**:
- `n_critics`: 5 (standard), 3 (faster), 7 (better uncertainty)
- `uncertainty_weight`: 0.1 (balanced), 0.01 (less conservative)

## Best Practices

1. **Start with base configurations** and modify incrementally
2. **Use inheritance** to avoid duplication
3. **Document changes** with inline comments
4. **Test hyperparameters** on smaller episode counts first
5. **Enable logging** (`use_wandb: true`) for experiment tracking
6. **Set appropriate seeds** for reproducibility
7. **Match batch size to available memory** (GPU/CPU)

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce `batch_size`
- Set `device: "cpu"`
- Reduce `hidden_size`

**Slow Training**:
- Increase `batch_size`
- Reduce `log_every` frequency
- Set `num_workers: 0` if multiprocessing issues

**Unstable Training**:
- Reduce learning rates
- Increase `tau` for slower target updates
- Enable gradient clipping (`grad_clip_norm`)

**Poor Performance**:
- Increase `episodes`
- Tune algorithm-specific parameters
- Check data normalization settings