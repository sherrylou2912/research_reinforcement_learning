# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an offline reinforcement learning framework implementing SAC (Soft Actor-Critic), CQL (Conservative Q-Learning), and SVRL (Structured Value-based Representation Learning) algorithms. The project focuses on learning from pre-collected datasets using Minari and D4RL datasets.

## Common Commands

### Installation
```bash
# Install with pip
pip install -r requirements.txt
pip install -e .

# Or with conda
conda env create -f environment.yml
conda activate offline-rl
```

### Training
```bash
# Basic training command
python train.py --config config/sac.yaml --agent sac --dataset 'mujoco/halfcheetah/medium-v0' --normalize_states --normalize_rewards

# Multiple seeds
python train.py --config config/cql.yaml --agent cql --dataset 'mujoco/halfcheetah/medium-v0' --normalize_states --normalize_rewards

# Multiple trials
python train.py --config config/svrl.yaml --agent svrl --dataset "D4RL/hopper/expert-v2" --num_trials 5
```

### Development
```bash
# No specific test or lint commands found - check with user for testing procedures
```

## Architecture

### Core Components
- **agent/**: Algorithm implementations (SAC, CQL, SVRL) with shared network architectures
- **data/**: Dataset loaders for Minari and D4RL with normalization support
- **config/**: YAML configuration files with inheritance via 'defaults' key
- **utils/**: Logging (with W&B integration), evaluation metrics, and rank utilities

### Key Design Patterns
1. **Configuration Inheritance**: Uses 'defaults' key in YAML configs to inherit from base.yaml
2. **Normalization**: Both states and rewards can be normalized via MinariDataset
3. **Multi-seed Training**: train.py supports running multiple seeds/trials automatically
4. **Device Management**: Automatic CUDA optimization when device='cuda'

### Agent Architecture
All agents (SAC/CQL/SVRL) follow the same interface:
- `__init__(state_size, action_size, config)`: Initialize with environment dimensions
- `update(batch)`: Single training step returning metrics dict
- `get_action(state)`: Policy inference
- `set_normalization_stats(stats)`: Apply dataset normalization
- `save(path)` / `load(path)`: Model persistence

### Data Flow
1. MinariDataset loads and preprocesses offline data
2. Normalization stats computed and applied to both data and agent
3. Training loop iterates over batches, calling agent.update()
4. Evaluation runs periodically using evaluate_policy()
5. Best models saved automatically based on evaluation returns

### Configuration System
- Base configurations in config/base.yaml
- Algorithm-specific configs inherit and override base settings
- Runtime arguments (dataset, normalization) passed via command line
- All hyperparameters accessible via config dict in agent constructors

## Important Notes
- Project uses Minari datasets (not D4RL directly)
- CUDA optimizations automatically applied when using GPU
- W&B logging optional via use_wandb config flag
- Models saved to models/ directory with naming convention: {agent}_{env}_seed{seed}_best.pt