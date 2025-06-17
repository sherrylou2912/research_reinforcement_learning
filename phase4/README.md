# Offline Reinforcement Learning Framework

This framework implements several state-of-the-art offline reinforcement learning algorithms, including SAC (Soft Actor-Critic), CQL (Conservative Q-Learning), and SVRL (Structured Value-based Representation Learning).

## Project Structure

```
phase4/
├── agent/                  # Agent implementations
│   ├── __init__.py
│   ├── sac.py             # SAC algorithm
│   ├── cql.py             # CQL algorithm
│   ├── svrl.py            # SVRL algorithm
│   ├── networks.py        # Neural network architectures
│   └── ua_lqe.py          # Uncertainty-Aware Low-rank Q Ensemble
├── completion/            # Matrix completion algorithms
│   └── soft_impute.py     # Soft imputation implementation
├── config/               # Configuration files
│   ├── sac.yaml          # SAC hyperparameters
│   ├── svrl.yaml         # SVRL hyperparameters
│   └── ua_lqe.yaml       # UA-LQE hyperparameters
├── data/                 # Data loading utilities
│   ├── buffer.py         # Replay buffer implementation
│   ├── d4rl_loader.py    # D4RL dataset loader
│   ├── minari_loader.py  # Minari dataset loader
│   └── test_data_loader.py
├── uncertainty/          # Uncertainty estimation methods
└── utils/               # Utility functions
    ├── eval_metrics.py   # Evaluation metrics
    ├── logger.py        # Logging utilities
    └── rank_utils.py    # Low-rank approximation utilities
```

## Components

### Agents

1. **SAC (Soft Actor-Critic)**
   - Base implementation with temperature auto-tuning
   - Supports state/reward normalization
   - Key parameters:
     * `gamma`: Discount factor (default: 0.99)
     * `tau`: Target network update rate (default: 0.005)
     * `learning_rate`: Learning rate (default: 3e-4)
     * `hidden_size`: Hidden layer size (default: 256)

2. **CQL (Conservative Q-Learning)**
   - Extends SAC with conservative value estimation
   - Additional parameters:
     * `cql_alpha`: CQL penalty coefficient (default: 1.0)
     * `cql_tau`: CQL temperature parameter (default: 10.0)
     * `with_lagrange`: Whether to use Lagrange multipliers (default: False)
     * `target_action_gap`: Target gap for Lagrange multipliers (default: 1.0)

3. **SVRL (Structured Value-based Representation Learning)**
   - Extends CQL with low-rank value function approximation
   - Additional parameters:
     * `rank`: Rank of the approximation (default: 10)
     * `svrl_weight`: Weight of SVRL loss (default: 1.0)
     * `soft_impute_iters`: Number of soft imputation iterations (default: 100)
     * `soft_impute_threshold`: Convergence threshold (default: 1e-4)

### Data Loading

1. **MinariDataset**
   - Supports loading from Minari datasets
   - Features:
     * Automatic data normalization
     * Batch processing
     * Statistics computation
   - Parameters:
     * `dataset_name`: Name of the Minari dataset
     * `normalize_states`: Whether to normalize states (default: True)
     * `normalize_rewards`: Whether to normalize rewards (default: True)
     * `device`: Device to store data on (default: 'cpu')
     * `download`: Whether to download dataset if not found (default: True)

2. **D4RLDataset**
   - Similar functionality for D4RL datasets
   - Compatible with standard D4RL benchmarks

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Basic Training Example

```python
import gymnasium as gym
from agent import SAC, CQL, SVRL
from data import create_minari_dataloader

# Create environment
env = gym.make('Pendulum-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# Configure agent
config = {
    'device': 'cuda',
    'gamma': 0.99,
    'tau': 0.005,
    'learning_rate': 3e-4,
    'hidden_size': 256,
    # For CQL
    'cql_alpha': 1.0,
    'cql_tau': 10.0,
    'with_lagrange': True,
    # For SVRL
    'rank': 10,
    'svrl_weight': 1.0
}

# Create agent (choose one)
agent = SAC(state_size, action_size, config)  # or CQL or SVRL

# Load dataset
data_loader, stats = create_minari_dataloader(
    dataset_name="D4RL/pen/expert-v2",
    batch_size=32,
    device=config['device']
)

# Set normalization statistics
agent.set_normalization_stats(stats)

# Training loop
for batch in data_loader:
    metrics = agent.update(batch)
    print(f"Training metrics: {metrics}")

# Save model
agent.save("model.pth")
```

### Loading a Trained Model

```python
# Load saved model
agent.load("model.pth")

# Evaluate
state = env.reset()[0]
while not done:
    action = agent.get_action(state)
    next_state, reward, done, _, info = env.step(action)
    state = next_state
```

## Configuration Files

The `config/` directory contains YAML files with default hyperparameters for each algorithm. Example for SAC:

```yaml
sac:
  gamma: 0.99
  tau: 0.005
  learning_rate: 0.0003
  hidden_size: 256
  batch_size: 256
  device: "cuda"
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 