"""
Training script for RL Memory Filter Agent

This script trains a PPO agent to select useful memories for WebArena tasks.

Usage:
    python train_rl_filter.py --train_tasks 0-100 --val_tasks 100-120 --epochs 50

Training Process:
1. Initialize PPO agent with MemoryFilterPolicy
2. For each epoch:
    a. Run WebArena tasks with RL-selected memories
    b. Collect trajectories and rewards (task success)
    c. Update policy using PPO
3. Save best model based on validation performance
4. Log training metrics to TensorBoard

Author: ARMPA Team  
Date: November 2025
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import pickle
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add webarena to path
sys.path.append(str(Path(__file__).parent.parent / "webarena"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

from memory.rl_filter_agent import MemoryFilterPolicy, MemoryFilterEnv, RLMemoryFilter
from memory.manager import MemoryManager

# Import WebArena components
from browser_env import ScriptBrowserEnv, create_stop_action
from agent import PromptAgent
from browser_env.helper_functions import RenderHelper


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback for logging training metrics.
    """
    
    def __init__(self, log_dir: Path, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    
                if 'task_success' in info:
                    self.episode_successes.append(float(info['task_success']))
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log statistics at end of rollout."""
        if self.episode_rewards:
            self.logger.record('train/mean_reward', np.mean(self.episode_rewards[-100:]))
            self.logger.record('train/mean_ep_length', np.mean(self.episode_lengths[-100:]))
            
        if self.episode_successes:
            self.logger.record('train/success_rate', np.mean(self.episode_successes[-100:]))


def create_webarena_env(
    config_file: str,
    memory_manager: MemoryManager,
    webarena_agent: PromptAgent,
    args: argparse.Namespace,
) -> MemoryFilterEnv:
    """
    Create a MemoryFilterEnv for a specific WebArena task.
    
    Args:
        config_file: Path to task config JSON
        memory_manager: MemoryManager for memory retrieval
        webarena_agent: WebArena agent for task execution
        args: Command-line arguments
    
    Returns:
        env: MemoryFilterEnv instance
    """
    env = MemoryFilterEnv(
        config_files=[config_file],
        memory_manager=memory_manager,
        webarena_agent=webarena_agent,
        max_steps=args.max_steps,
        embedding_dim=args.embedding_dim,
        max_memories=args.max_memories,
    )
    return env


def train_rl_filter(args: argparse.Namespace):
    """
    Main training function for RL memory filter agent.
    
    Args:
        args: Command-line arguments
    """
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"rl_filter_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Training RL Memory Filter Agent")
    print(f"{'='*80}")
    print(f"Output directory: {run_dir}")
    print(f"Training tasks: {args.train_start}-{args.train_end}")
    print(f"Validation tasks: {args.val_start}-{args.val_end}")
    print(f"Total epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Save training config
    config = vars(args)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize memory manager
    print("Initializing MemoryManager...")
    memory_manager = MemoryManager(collection_name=args.memory_collection)
    
    # Initialize WebArena agent (for task execution)
    print("Initializing WebArena agent...")
    webarena_agent = PromptAgent(
        action_set_tag=args.action_set_tag,
        model=args.model,
        temperature=args.temperature,
        use_litellm=True,
        instruction_path=args.instruction_path,
    )
    
    # Get training task config files
    config_dir = Path("webarena/config_files")
    train_configs = [
        str(config_dir / f"{i}.json")
        for i in range(args.train_start, args.train_end)
        if (config_dir / f"{i}.json").exists()
    ]
    
    val_configs = [
        str(config_dir / f"{i}.json")
        for i in range(args.val_start, args.val_end)
        if (config_dir / f"{i}.json").exists()
    ]
    
    print(f"Found {len(train_configs)} training tasks")
    print(f"Found {len(val_configs)} validation tasks")
    
    if len(train_configs) == 0:
        raise ValueError("No training tasks found!")
    
    # Create training environment
    print("\nCreating training environment...")
    
    def make_env():
        return MemoryFilterEnv(
            config_files=train_configs,
            memory_manager=memory_manager,
            webarena_agent=webarena_agent,
            max_steps=args.max_steps,
            embedding_dim=args.embedding_dim,
            max_memories=args.max_memories,
        )
    
    # Vectorize environment
    env = DummyVecEnv([make_env])
    
    # Create validation environment if validation tasks exist
    eval_env = None
    if len(val_configs) > 0:
        def make_eval_env():
            return MemoryFilterEnv(
                config_files=val_configs,
                memory_manager=memory_manager,
                webarena_agent=webarena_agent,
                max_steps=args.max_steps,
                embedding_dim=args.embedding_dim,
                max_memories=args.max_memories,
            )
        eval_env = DummyVecEnv([make_eval_env])
    
    # Initialize PPO model
    print("\nInitializing PPO model...")
    
    # Custom policy kwargs to use our MemoryFilterPolicy architecture
    policy_kwargs = dict(
        net_arch=[args.hidden_dim, args.hidden_dim],
        activation_fn=torch.nn.ReLU,
    )
    
    model = PPO(
        policy="MlpPolicy",  # We'll customize this
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=args.device,
        tensorboard_log=str(log_dir),
    )
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="rl_filter",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Training metrics callback
    metrics_callback = TrainingMetricsCallback(log_dir=log_dir)
    callbacks.append(metrics_callback)
    
    # Evaluation callback
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(log_dir / "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
    
    # Configure logger
    new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Train the model
    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=args.log_interval,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training summary
    summary = {
        'timestamp': timestamp,
        'total_timesteps': args.total_timesteps,
        'num_train_tasks': len(train_configs),
        'num_val_tasks': len(val_configs),
        'final_model_path': str(final_model_path),
    }
    
    if len(metrics_callback.episode_successes) > 0:
        summary['final_success_rate'] = float(np.mean(metrics_callback.episode_successes[-100:]))
        summary['final_mean_reward'] = float(np.mean(metrics_callback.episode_rewards[-100:]))
    
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Results saved to: {run_dir}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {log_dir}")
    
    return model, run_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL Memory Filter Agent for WebArena"
    )
    
    # Task configuration
    parser.add_argument("--train_start", type=int, default=0,
                       help="Start index for training tasks")
    parser.add_argument("--train_end", type=int, default=100,
                       help="End index for training tasks")
    parser.add_argument("--val_start", type=int, default=100,
                       help="Start index for validation tasks")
    parser.add_argument("--val_end", type=int, default=120,
                       help="End index for validation tasks")
    
    # Memory configuration
    parser.add_argument("--memory_collection", type=str, default="webarena",
                       help="Qdrant collection name for memories")
    parser.add_argument("--embedding_dim", type=int, default=384,
                       help="Embedding dimension")
    parser.add_argument("--max_memories", type=int, default=10,
                       help="Maximum number of memories to retrieve")
    
    # WebArena agent configuration
    parser.add_argument("--action_set_tag", type=str, default="id_accessibility_tree",
                       help="Action set tag for WebArena")
    parser.add_argument("--model", type=str,
                       default="together_ai/Qwen/Qwen2.5-72B-Instruct",
                       help="LLM model for WebArena agent")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for LLM")
    parser.add_argument("--instruction_path", type=str,
                       default="agent/prompts/raw/p_cot_id_actree_2s_no_na_memory.py",
                       help="Path to instruction prompt")
    parser.add_argument("--max_steps", type=int, default=30,
                       help="Maximum steps per task")
    
    # RL training hyperparameters
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--total_timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate for PPO")
    parser.add_argument("--n_steps", type=int, default=2048,
                       help="Steps per rollout")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=10,
                       help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda")
    parser.add_argument("--clip_range", type=float, default=0.2,
                       help="PPO clip range")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                       help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                       help="Max gradient norm")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension for policy network")
    
    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, default="rl_training_runs",
                       help="Directory for training outputs")
    parser.add_argument("--save_freq", type=int, default=10000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_freq", type=int, default=5000,
                       help="Evaluate every N steps")
    parser.add_argument("--n_eval_episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log interval (in updates)")
    
    # Device
    parser.add_argument("--device", type=str, default="mps",
                       help="Device for training (mps, cuda, cpu)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Check device availability
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Run training
    model, run_dir = train_rl_filter(args)
