import gym
import torch
from torch.utils.data import TensorDataset, DataLoader


class D4rlDataset:
    def __init__(self, env_id="halfcheetah-medium-v2", batch_size=256, seed=1, eval_env_id=None):
        self.env_id = env_id
        self.batch_size = batch_size
        self.seed = seed
        self.eval_env_id = eval_env_id or self._get_eval_env_id(env_id)
        self.dataloader = None
        self.eval_env = None
        
    def _get_eval_env_id(self, env_id):
        """Map dataset environment to evaluation environment."""
        env_mapping = {
            "halfcheetah-medium-v2": "HalfCheetah-v2",
            "halfcheetah-expert-v2": "HalfCheetah-v2",
            "halfcheetah-random-v2": "HalfCheetah-v2",
            "walker2d-medium-v2": "Walker2d-v2",
            "walker2d-expert-v2": "Walker2d-v2",
            "hopper-medium-v2": "Hopper-v2",
            "hopper-expert-v2": "Hopper-v2",
            # Add more mappings as needed
        }
        
        # Try exact match first
        if env_id in env_mapping:
            return env_mapping[env_id]
        
        # Fallback: extract base environment name
        for dataset_env, eval_env in env_mapping.items():
            if dataset_env.split('-')[0] in env_id:
                return eval_env
        
        # Default: try to convert dataset env to standard env
        base_name = env_id.split('-')[0]
        return f"{base_name.title()}-v2"
        
    def prep_dataloader(self):
        """Prepare the dataloader and evaluation environment."""
        env = gym.make(self.env_id)
        dataset = env.get_dataset()
        tensors = {}
        
        for k, v in dataset.items():
            if k in ["actions", "observations", "next_observations", "rewards", "terminals"]:
                if k != "terminals":
                    tensors[k] = torch.from_numpy(v).float()
                else:
                    tensors[k] = torch.from_numpy(v).long()
        
        tensordata = TensorDataset(
            tensors["observations"],
            tensors["actions"],
            tensors["rewards"][:, None],
            tensors["next_observations"],
            tensors["terminals"][:, None]
        )
        
        self.dataloader = DataLoader(tensordata, batch_size=self.batch_size, shuffle=True)
        self._prepare_eval_env()
        
        return self.dataloader, self.eval_env
    
    def _prepare_eval_env(self):
        """Prepare the evaluation environment with error handling."""
        try:
            self.eval_env = gym.make(self.eval_env_id)
            self.eval_env.seed(self.seed)
        except gym.error.UnregisteredEnv:
            print(f"Warning: Could not create eval environment '{self.eval_env_id}'. "
                  f"Falling back to dataset environment '{self.env_id}'")
            self.eval_env = gym.make(self.env_id)
            self.eval_env.seed(self.seed)
        except Exception as e:
            print(f"Error creating eval environment: {e}")
            self.eval_env = None
    
    def get_dataloader(self):
        """Get the prepared dataloader. If not prepared yet, prepare it first."""
        if self.dataloader is None:
            self.prep_dataloader()
        return self.dataloader
    
    def get_eval_env(self):
        """Get the evaluation environment. If not prepared yet, prepare it first."""
        if self.eval_env is None:
            self.prep_dataloader()
        return self.eval_env
    
    def reset(self):
        """Reset the module to allow re-preparation with different parameters."""
        self.dataloader = None
        self.eval_env = None


# Usage example:
if __name__ == "__main__":
    # Initialize the module (automatic eval env mapping)
    data_module = D4rlDataset(env_id="halfcheetah-medium-v2", batch_size=256, seed=1)
    
    # Or specify custom eval environment
    data_module_custom = D4rlDataset(
        env_id="halfcheetah-medium-v2", 
        batch_size=256, 
        seed=1,
        eval_env_id="HalfCheetah-v3"  # Custom eval environment
    )
    
    # Prepare dataloader and eval environment
    dataloader, eval_env = data_module.prep_dataloader()
    
    # Or get them individually
    # dataloader = data_module.get_dataloader()
    # eval_env = data_module.get_eval_env()