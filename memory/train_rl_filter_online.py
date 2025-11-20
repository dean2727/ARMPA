"""
Online RL Training Script for Memory Filter Agent

This script trains the memory filter using online, on-policy RL. It alternates between:
1. Collecting episodes using the current filter policy
2. Updating the policy using GRPO on the collected episodes

The process repeats until convergence or max cycles reached.

Usage:
    python memory/train_rl_filter_online.py \
        --model_dir models/rl_filter \
        --num_cycles 20 \
        --tasks_per_cycle 10 \
        --model "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo"

Author: ARMPA Research Team
Date: 2025-11-19
"""

import argparse
import pickle
import logging
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from memory.rl_filter_agent import RLMemoryFilter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_episodes_with_filter(
    rl_filter: Optional[RLMemoryFilter],
    num_tasks: int,
    model: str,
    instruction_path: str,
    temperature: float = 0.7,
    num_memories: int = 10,
    rl_filter_threshold: float = 0.5,
    temp_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Collect episodes using the current filter policy.
    
    Args:
        rl_filter: Current RL filter agent (None for first cycle)
        num_tasks: Number of tasks to run
        model: LLM model to use
        instruction_path: Path to prompt template
        temperature: Sampling temperature
        num_memories: Max memories to retrieve
        rl_filter_threshold: Gate threshold for selection
        temp_dir: Temporary directory for episode buffers
    
    Returns:
        episodes: List of episode dictionaries with recall events and rewards
    """
    if temp_dir is None:
        temp_dir = Path(f"runs/online_rl_{int(time.time())}")
    
    # Convert to absolute path to avoid issues when changing cwd
    temp_dir = temp_dir.resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save current filter model if it exists
    filter_model_path = None
    if rl_filter is not None:
        filter_model_path = temp_dir / "current_filter.pt"
        rl_filter.save_model(str(filter_model_path))
    
    # Build command to run WebArena tasks (will run from webarena directory)
    cmd = [
        sys.executable,
        "run.py",  # Changed from webarena/run.py since we'll run from webarena dir
        "--agent_type", "litellm",
        "--instruction_path", f"agent/prompts/raw/{Path(instruction_path).name}",
        "--model", model,
        "--temperature", str(temperature),
        "--get_memory",
        "--num_memories", str(num_memories),
        "--collect_rl_data",
        "--num_tasks", str(num_tasks),
        "--result_dir", str(temp_dir / "results"),
    ]
    
    # Add filter arguments if we have a trained filter
    if filter_model_path is not None:
        cmd.extend([
            "--use_rl_filter",
            "--rl_filter_model", str(filter_model_path),
            "--rl_filter_threshold", str(rl_filter_threshold),
        ])
    
    # Set environment variable for PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + ":" + env.get("PYTHONPATH", "")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 Collecting {num_tasks} episodes with current policy...")
    logger.info(f"{'='*70}")
    
    # Run the command from webarena directory (needed for config_files path)
    webarena_dir = Path.cwd() / "webarena"
    
    # Create a progress bar for episode collection
    pbar = tqdm(total=num_tasks, desc="Collecting episodes", unit="task", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=webarena_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        
        # Stream output and update progress bar
        for line in process.stdout:
            if "Task" in line and "completed" in line:
                pbar.update(1)
        
        # Wait for completion
        process.wait(timeout=3600)
        pbar.close()
        
        if process.returncode != 0:
            stderr = process.stderr.read() if process.stderr else ""
            logger.error(f"❌ Episode collection failed with return code {process.returncode}")
            logger.error(f"STDERR: {stderr}")
            raise RuntimeError("Episode collection failed")
        
        logger.info("✅ Episode collection completed successfully")
        
    except subprocess.TimeoutExpired:
        logger.error("Episode collection timed out after 1 hour")
        raise
    
    # Load collected episodes from webarena/runs/<timestamp>/episode_buffers/
    # (run.py saves to runs/ directory, not result_dir)
    webarena_runs_dir = Path.cwd() / "webarena" / "runs"
    
    # Find the most recent run directory
    run_dirs = sorted([d for d in webarena_runs_dir.iterdir() if d.is_dir()], 
                      key=lambda x: x.stat().st_mtime, reverse=True)
    
    episode_dir = None
    for run_dir in run_dirs:
        potential_dir = run_dir / "episode_buffers"
        if potential_dir.exists():
            episode_dir = potential_dir
            break
    
    if episode_dir is None or not episode_dir.exists():
        logger.error(f"Episode buffer directory not found in {webarena_runs_dir}")
        logger.error(f"Checked directories: {[str(d) for d in run_dirs[:3]]}")
        raise FileNotFoundError(f"No episode buffers found")
    
    logger.info(f"Loading episodes from: {episode_dir}")
    episode_files = list(episode_dir.glob("*.pkl"))
    logger.info(f"Found {len(episode_files)} episode files")
    
    episodes = []
    for ep_file in episode_files:
        try:
            with open(ep_file, 'rb') as f:
                episode = pickle.load(f)
            
            # Validate episode structure
            if 'recall_events' in episode and 'final_reward' in episode:
                episodes.append(episode)
            else:
                logger.warning(f"Skipping invalid episode: {ep_file.name}")
        except Exception as e:
            logger.warning(f"Error loading {ep_file.name}: {e}")
    
    logger.info(f"Loaded {len(episodes)} valid episodes")
    return episodes


def evaluate_policy(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate policy performance on collected episodes.
    
    Args:
        episodes: List of episode dictionaries
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if not episodes:
        return {}
    
    successes = sum(1 for ep in episodes if ep.get('success', False))
    total_steps = sum(ep.get('num_steps', 0) for ep in episodes)
    rewards = [ep.get('final_reward', 0.0) for ep in episodes]
    
    # Count recall events and memory selection
    total_recall_events = 0
    total_candidates = 0
    total_selected = 0
    
    for ep in episodes:
        for recall_event in ep.get('recall_events', []):
            total_recall_events += 1
            candidates = recall_event.get('candidates', [])
            total_candidates += len(candidates)
            total_selected += sum(1 for c in candidates if c.get('selected', False))
    
    metrics = {
        'num_episodes': len(episodes),
        'success_rate': successes / len(episodes) if episodes else 0.0,
        'avg_steps': total_steps / len(episodes) if episodes else 0.0,
        'avg_reward': np.mean(rewards) if rewards else 0.0,
        'std_reward': np.std(rewards) if rewards else 0.0,
        'total_recall_events': total_recall_events,
        'avg_candidates_per_recall': total_candidates / max(total_recall_events, 1),
        'avg_selected_per_recall': total_selected / max(total_recall_events, 1),
        'selection_rate': total_selected / max(total_candidates, 1),
    }
    
    return metrics


def train_online_rl(
    rl_filter: RLMemoryFilter,
    num_cycles: int,
    tasks_per_cycle: int,
    model: str,
    instruction_path: str,
    temperature: float,
    num_memories: int,
    rl_filter_threshold: float,
    model_dir: Path,
    convergence_threshold: float = 0.01,
    patience: int = 3,
    disable_early_stopping: bool = False,
) -> None:
    """
    Train RL filter using online, on-policy learning.
    
    Args:
        rl_filter: RL filter agent to train
        num_cycles: Maximum number of rollout-update cycles
        tasks_per_cycle: Number of tasks to collect per cycle
        model: LLM model name
        instruction_path: Path to prompt template
        temperature: Sampling temperature
        num_memories: Max memories to retrieve
        rl_filter_threshold: Gate threshold
        model_dir: Directory to save model checkpoints
        convergence_threshold: Stop if reward improvement < this
        patience: Number of cycles without improvement before stopping
        disable_early_stopping: If True, train for all num_cycles regardless of convergence
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    log_dir = model_dir / "tensorboard_logs"
    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"📊 TensorBoard logs: {log_dir}")
    logger.info(f"   Run: tensorboard --logdir {log_dir}")
    
    best_reward = -float('inf')
    no_improvement_count = 0
    
    logger.info(f"\n🚀 Starting online RL training for {num_cycles} cycles")
    logger.info(f"   Tasks per cycle: {tasks_per_cycle}")
    logger.info(f"   Total episodes: ~{num_cycles * tasks_per_cycle}")
    
    # Create progress bar for cycles
    cycle_pbar = tqdm(range(num_cycles), desc="Training Progress", 
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} cycles [{elapsed}<{remaining}]')
    
    for cycle in cycle_pbar:
        cycle_pbar.set_description(f"🔄 Cycle {cycle + 1}/{num_cycles}")
        
        # Step 1: Collect episodes with current policy
        episodes = collect_episodes_with_filter(
            rl_filter=rl_filter if cycle > 0 else None,  # Use random for first cycle
            num_tasks=tasks_per_cycle,
            model=model,
            instruction_path=instruction_path,
            temperature=temperature,
            num_memories=num_memories,
            rl_filter_threshold=rl_filter_threshold,
            temp_dir=model_dir / f"cycle_{cycle}",
        )
        
        if not episodes:
            logger.error(f"No episodes collected in cycle {cycle}, stopping")
            break
        
        # Step 2: Evaluate policy
        metrics = evaluate_policy(episodes)
        
        # Log to TensorBoard
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"metrics/{key}", value, cycle)
        
        logger.info(f"\n📈 Cycle {cycle + 1} Metrics:")
        logger.info(f"   Success Rate: {metrics['success_rate']:.1%}")
        logger.info(f"   Avg Reward: {metrics['avg_reward']:.4f}")
        logger.info(f"   Selection Rate: {metrics['selection_rate']:.1%}")
        logger.info(f"   Avg Steps: {metrics['avg_steps']:.1f}")
        
        # Step 3: Update policy using GRPO
        logger.info(f"\n🔧 Updating policy with {len(episodes)} episodes...")
        update_metrics = rl_filter.update_policy_grpo(episodes=episodes)
        
        # Log update metrics to TensorBoard
        for key, value in update_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"training/{key}", value, cycle)
        
        logger.info(f"   Loss: {update_metrics.get('loss', 0):.4f}")
        logger.info(f"   KL Div: {update_metrics.get('kl_div', 0):.4f}")
        
        # Step 4: Save checkpoint
        checkpoint_path = model_dir / f"checkpoint_cycle_{cycle + 1}.pt"
        rl_filter.save_model(str(checkpoint_path))
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Step 5: Check for improvement and convergence
        current_reward = metrics['avg_reward']
        improvement = current_reward - best_reward
        
        if improvement > convergence_threshold:
            best_reward = current_reward
            no_improvement_count = 0
            best_model_path = model_dir / "best_model.pt"
            rl_filter.save_model(str(best_model_path))
            logger.info(f"✓ New best model saved (reward: {best_reward:.4f}, improvement: {improvement:.4f})")
        # Step 4: Save checkpoint
        checkpoint_path = model_dir / f"checkpoint_cycle_{cycle + 1}.pt"
        rl_filter.save_model(str(checkpoint_path))
        
        # Step 5: Check for improvement and convergence
        current_reward = metrics['avg_reward']
        improvement = current_reward - best_reward
        
        writer.add_scalar("training/best_reward", best_reward, cycle)
        writer.add_scalar("training/improvement", improvement, cycle)
        
        if improvement > convergence_threshold:
            best_reward = current_reward
            no_improvement_count = 0
            best_model_path = model_dir / "best_model.pt"
            rl_filter.save_model(str(best_model_path))
            logger.info(f"\n✨ New best model! Reward: {best_reward:.4f} (↑ {improvement:.4f})")
        else:
            no_improvement_count += 1
            logger.info(f"   No improvement (best: {best_reward:.4f}, patience: {no_improvement_count}/{patience})")
        
        # Check convergence (only if early stopping is enabled)
        if not disable_early_stopping and no_improvement_count >= patience:
            logger.info(f"\n⚠️  Early stopping: No improvement for {patience} cycles")
            cycle_pbar.close()
            break
    
    cycle_pbar.close()
    writer.close()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🎉 Training completed!")
    logger.info(f"{'='*70}")
    logger.info(f"   Best reward: {best_reward:.4f}")
    logger.info(f"   Best model: {model_dir / 'best_model.pt'}")
    logger.info(f"   TensorBoard: tensorboard --logdir {log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train RL Memory Filter with Online RL")
    
    # Model arguments
    parser.add_argument("--model_dir", type=str, default="models/rl_filter_online",
                       help="Directory to save model checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    
    # Online RL arguments
    parser.add_argument("--num_cycles", type=int, default=20,
                       help="Number of rollout-update cycles")
    parser.add_argument("--tasks_per_cycle", type=int, default=10,
                       help="Number of tasks to collect per cycle (processed sequentially)")
    parser.add_argument("--convergence_threshold", type=float, default=0.01,
                       help="Stop if reward improvement < this")
    parser.add_argument("--patience", type=int, default=3,
                       help="Cycles without improvement before stopping")
    parser.add_argument("--disable_early_stopping", action="store_true",
                       help="If set, train for all num_cycles regardless of convergence")
    
    # WebArena task arguments
    parser.add_argument("--model", type=str, 
                       default="together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
                       help="LLM model to use")
    parser.add_argument("--instruction_path", type=str,
                       default="webarena/agent/prompts/raw/p_cot_id_actree_2s_no_na_memory.py",
                       help="Path to prompt template")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--num_memories", type=int, default=10,
                       help="Max number of memories to retrieve")
    
    # RL filter arguments
    parser.add_argument("--rl_filter_threshold", type=float, default=0.5,
                       help="Gate score threshold for memory selection")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate for optimizer")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                       help="PPO clipping parameter")
    parser.add_argument("--kl_beta", type=float, default=0.01,
                       help="KL divergence penalty coefficient")
    parser.add_argument("--gamma", type=float, default=0.5,
                       help="Reward shaping parameter for step efficiency")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda", "mps"],
                       help="Device for training")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, using CPU")
        args.device = "cpu"
    
    logger.info(f"Using device: {args.device}")
    
    # Initialize RL filter
    logger.info("Initializing RL filter...")
    rl_filter = RLMemoryFilter(
        task_dim=1024,
        obs_dim=1024,
        memory_dim=1024,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        kl_beta=args.kl_beta,
        gamma=args.gamma,
        score_threshold=args.rl_filter_threshold,
        device=args.device,
        model_path=args.resume_from,
    )
    
    model_dir = Path(args.model_dir)
    
    # Start online RL training
    train_online_rl(
        rl_filter=rl_filter,
        num_cycles=args.num_cycles,
        tasks_per_cycle=args.tasks_per_cycle,
        model=args.model,
        instruction_path=args.instruction_path,
        temperature=args.temperature,
        num_memories=args.num_memories,
        rl_filter_threshold=args.rl_filter_threshold,
        model_dir=model_dir,
        convergence_threshold=args.convergence_threshold,
        patience=args.patience,
        disable_early_stopping=args.disable_early_stopping,
    )


if __name__ == "__main__":
    main()
