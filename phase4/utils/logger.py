import wandb
import numpy as np
from typing import Dict, Any, Optional
import os
import json
from datetime import datetime

class Logger:
    def __init__(
        self,
        project_name: str,
        config: Dict[str, Any],
        output_dir: str = "logs",
        use_wandb: bool = True
    ):
        """
        Initialize logger for experiment tracking.
        
        Args:
            project_name: Name of the project
            config: Configuration dictionary
            output_dir: Directory to save logs
            use_wandb: Whether to use Weights & Biases
        """
        self.use_wandb = use_wandb
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project=project_name,
                config=config,
                dir=output_dir
            )
        
        # Save config
        self.config = config
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            
        # Initialize metrics
        self.metrics = {}
        
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics for the current step.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step number (optional)
        """
        # Update internal metrics
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(metrics, step=step)
            
        # Save metrics to file
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
            
    def log_model(
        self,
        model: Any,
        name: str
    ):
        """
        Save a model checkpoint.
        
        Args:
            model: Model to save
            name: Name of the checkpoint
        """
        if self.use_wandb:
            wandb.save(name)
            
        # Save locally
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        path = os.path.join(checkpoint_dir, f"{name}.pt")
        model.save(path)
        
    def finish(self):
        """Clean up and finish logging."""
        if self.use_wandb:
            wandb.finish()
            
    def get_metric_statistics(
        self,
        metric_name: str
    ) -> Dict[str, float]:
        """
        Get statistics for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary containing mean, std, min, max values
        """
        if metric_name not in self.metrics:
            return {}
            
        values = np.array(self.metrics[metric_name])
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

if __name__ == "__main__":
    # 测试日志记录器
    print("Testing logger functionality...")
    
    # 创建测试配置
    config = {
        'algorithm': 'SAC',
        'env_name': 'Pendulum-v1',
        'seed': 42,
        'batch_size': 256,
        'learning_rate': 3e-4
    }
    
    # 创建日志记录器
    logger = Logger(
        project_name="test_experiment",
        config=config,
        output_dir="./test_logs",
        use_wandb=False
    )
    
    print("\nLogger initialized with config:")
    print(json.dumps(config, indent=2))
    
    # 测试记录标量
    for step in range(10):
        metrics = {
            'train/q1_loss': np.random.rand(),
            'train/q2_loss': np.random.rand(),
            'train/policy_loss': np.random.rand(),
            'train/alpha_loss': np.random.rand(),
            'eval/average_return': np.random.normal(loc=-100 + step*10, scale=5)
        }
        
        logger.log_metrics(metrics, step)
        
        if step % 3 == 0:
            print(f"\nLogged metrics at step {step}:")
            print(json.dumps(metrics, indent=2))
    
    # 测试记录直方图
    values = np.random.randn(1000)
    logger.log_histogram('test/value_distribution', values, step=10)
    print("\nLogged histogram data")
    
    # 测试记录图像
    image = np.random.rand(32, 32, 3)  # 随机RGB图像
    logger.log_image('test/random_image', image, step=10)
    print("\nLogged test image")
    
    # 测试保存和加载检查点
    state = {
        'step': 10,
        'model_state': {'weight': np.random.rand(10, 10)},
        'optimizer_state': {'momentum': np.random.rand(10, 10)}
    }
    
    logger.log_model(state, 'test_checkpoint.pth')
    print("\nSaved checkpoint")
    
    loaded_state = logger.log_model(state, 'test_checkpoint.pth')
    print("Loaded checkpoint")
    
    # 验证加载的检查点
    assert loaded_state['step'] == state['step'], "Checkpoint loading failed!"
    
    # 测试结束
    logger.finish()
    print("\nAll logger tests passed!")
    
    # 清理测试文件
    import shutil
    if os.path.exists("./test_logs"):
        shutil.rmtree("./test_logs")
    print("\nCleaned up test files")
