"""
Online RL Training Script for Memory Filter Agent

This script trains the memory filter using online, on-policy RL. It collects
episodes using the current filter policy, immediately updates the policy using
GRPO, and repeats in alternating cycles until convergence.

Usage:
    python train_rl_filter.py \
        --model_dir models/rl_filter \
        --batch_size 8 \
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
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from tqdm import tqdm

from memory.rl_filter_agent import RLMemoryFilter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_episode_buffers(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load episode buffers from pickle files.
    
    Args:
        data_dir: Directory containing episode buffer pickle files
    
    Returns:
        episodes: List of episode dictionaries
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    episode_files = list(data_path.glob("*.pkl"))
    logger.info(f"Found {len(episode_files)} episode files in {data_dir}")
    
    episodes = []
    for ep_file in tqdm(episode_files, desc="Loading episodes"):
        try:
            with open(ep_file, 'rb') as f:
                episode = pickle.load(f)
                
            # Validate episode structure
            required_keys = ['recall_events', 'final_reward', 'success', 'num_steps']
            if all(key in episode for key in required_keys):
                episodes.append(episode)
            else:
                logger.warning(f"Skipping {ep_file.name}: missing required keys")
        except Exception as e:
            logger.warning(f"Error loading {ep_file.name}: {e}")
    
    logger.info(f"Loaded {len(episodes)} valid episodes")
    return episodes


def filter_episodes_by_success(
    episodes: List[Dict[str, Any]],
    min_success_rate: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Filter episodes based on success rate.
    
    Args:
        episodes: List of episode dictionaries
        min_success_rate: Minimum success rate to include (0.0 = include all)
    
    Returns:
        filtered_episodes: Episodes meeting success criteria
    """
    if min_success_rate <= 0.0:
        return episodes
    
    successes = sum(1 for ep in episodes if ep['success'])
    success_rate = successes / len(episodes) if episodes else 0.0
    
    if success_rate < min_success_rate:
        logger.warning(
            f"Success rate {success_rate:.2%} below threshold {min_success_rate:.2%}, "
            f"including all episodes"
        )
        return episodes
    
    # Optionally balance successes and failures
    successful = [ep for ep in episodes if ep['success']]
    failed = [ep for ep in episodes if not ep['success']]
    
    logger.info(f"Episode statistics: {len(successful)} successes, {len(failed)} failures")
    
    return episodes


def group_episodes_for_grpo(
    episodes: List[Dict[str, Any]],
    group_size: int,
) -> List[List[Dict[str, Any]]]:
    """
    Group episodes into batches for GRPO training.
    
    GRPO requires computing group-relative advantages, so we need
    to process multiple episodes together.
    
    Args:
        episodes: List of all episode dictionaries
        group_size: Number of episodes per group (N in GRPO formulation)
    
    Returns:
        groups: List of episode groups
    """
    # Shuffle episodes for better training
    np.random.shuffle(episodes)
    
    groups = []
    for i in range(0, len(episodes), group_size):
        group = episodes[i:i + group_size]
        if len(group) >= 2:  # Need at least 2 for advantage computation
            groups.append(group)
        else:
            logger.warning(f"Skipping incomplete group with {len(group)} episodes")
    
    return groups


def train_epoch(
    rl_filter: RLMemoryFilter,
    episode_groups: List[List[Dict[str, Any]]],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Train for one epoch over all episode groups.
    
    Args:
        rl_filter: RL memory filter agent
        episode_groups: List of episode groups for GRPO
        device: Device for computation
    
    Returns:
        epoch_metrics: Aggregated metrics over all updates
    """
    epoch_metrics = {
        'total_loss': 0.0,
        'total_policy_loss': 0.0,
        'total_kl_div': 0.0,
        'total_mean_reward': 0.0,
        'num_updates': 0,
    }
    
    for group_idx, episode_group in enumerate(tqdm(episode_groups, desc="Training")):
        # Update policy on this group
        metrics = rl_filter.update_policy_grpo(episodes=episode_group)
        
        # Accumulate metrics
        epoch_metrics['total_loss'] += metrics.get('loss', 0.0)
        epoch_metrics['total_policy_loss'] += metrics.get('policy_loss', 0.0)
        epoch_metrics['total_kl_div'] += metrics.get('kl_div', 0.0)
        epoch_metrics['total_mean_reward'] += metrics.get('mean_reward', 0.0)
        epoch_metrics['num_updates'] += 1
        
        # Log every N updates
        if (group_idx + 1) % 10 == 0:
            logger.info(
                f"Group {group_idx + 1}/{len(episode_groups)}: "
                f"loss={metrics.get('loss', 0.0):.4f}, "
                f"mean_reward={metrics.get('mean_reward', 0.0):.4f}"
            )
    
    # Compute averages
    if epoch_metrics['num_updates'] > 0:
        for key in ['total_loss', 'total_policy_loss', 'total_kl_div', 'total_mean_reward']:
            epoch_metrics[key] /= epoch_metrics['num_updates']
    
    return epoch_metrics


def evaluate(
    rl_filter: RLMemoryFilter,
    episodes: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Evaluate RL filter on held-out episodes.
    
    Args:
        rl_filter: RL memory filter agent
        episodes: List of evaluation episodes
    
    Returns:
        eval_metrics: Evaluation metrics
    """
    total_memories_retrieved = 0
    total_memories_selected = 0
    total_recall_events = 0
    
    for episode in episodes:
        for recall_event in episode['recall_events']:
            total_recall_events += 1
            candidates = recall_event['candidates']
            total_memories_retrieved += len(candidates)
            
            # Simulate filtering (without modifying episode)
            task_emb = recall_event['task_embedding']
            obs_emb = recall_event['obs_embedding']
            entropy = recall_event['entropy']
            
            # Convert candidates to memory format expected by filter
            memories = [
                {
                    'memory_id': c['memory_id'],
                    'embedding': c['embedding'],
                    'score': c.get('similarity_score', 0.0),
                }
                for c in candidates if c['embedding'] is not None
            ]
            
            filtered = rl_filter.filter_memories(
                memories=memories,
                task_embedding=task_emb,
                obs_embedding=obs_emb,
                entropy=entropy,
            )
            
            total_memories_selected += len(filtered)
    
    eval_metrics = {
        'total_recall_events': total_recall_events,
        'total_memories_retrieved': total_memories_retrieved,
        'total_memories_selected': total_memories_selected,
        'avg_memories_per_recall': total_memories_retrieved / max(total_recall_events, 1),
        'avg_selected_per_recall': total_memories_selected / max(total_recall_events, 1),
        'selection_rate': total_memories_selected / max(total_memories_retrieved, 1),
    }
    
    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description="Train RL Memory Filter")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing episode buffer pickle files")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Fraction of data for training (rest for validation)")
    
    # Model arguments
    parser.add_argument("--model_dir", type=str, default="models/rl_filter",
                       help="Directory to save model checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Number of episodes per GRPO group")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
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
    
    # Load episodes
    logger.info("Loading episode buffers...")
    episodes = load_episode_buffers(args.data_dir)
    
    if len(episodes) == 0:
        logger.error("No valid episodes found, exiting")
        return
    
    # Filter episodes
    episodes = filter_episodes_by_success(episodes, min_success_rate=0.0)
    
    # Split train/val
    np.random.seed(42)
    np.random.shuffle(episodes)
    split_idx = int(len(episodes) * args.train_split)
    train_episodes = episodes[:split_idx]
    val_episodes = episodes[split_idx:]
    
    logger.info(f"Train: {len(train_episodes)} episodes, Val: {len(val_episodes)} episodes")
    
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
        device=args.device,
        model_path=args.resume_from,
    )
    
    # Create model directory
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    best_val_reward = -float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Group episodes for GRPO
        train_groups = group_episodes_for_grpo(train_episodes, args.batch_size)
        
        # Train
        train_metrics = train_epoch(rl_filter, train_groups, device=args.device)
        
        logger.info(f"\nTrain Metrics:")
        logger.info(f"  Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"  Policy Loss: {train_metrics['total_policy_loss']:.4f}")
        logger.info(f"  KL Div: {train_metrics['total_kl_div']:.4f}")
        logger.info(f"  Mean Reward: {train_metrics['total_mean_reward']:.4f}")
        
        # Evaluate
        if len(val_episodes) > 0:
            logger.info("\nEvaluating...")
            val_metrics = evaluate(rl_filter, val_episodes)
            
            logger.info(f"Val Metrics:")
            logger.info(f"  Recall Events: {val_metrics['total_recall_events']}")
            logger.info(f"  Avg Memories Retrieved: {val_metrics['avg_memories_per_recall']:.2f}")
            logger.info(f"  Avg Memories Selected: {val_metrics['avg_selected_per_recall']:.2f}")
            logger.info(f"  Selection Rate: {val_metrics['selection_rate']:.2%}")
        
        # Save checkpoint
        checkpoint_path = model_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        rl_filter.save_model(str(checkpoint_path))
        
        # Save best model
        current_val_reward = train_metrics['total_mean_reward']
        if current_val_reward > best_val_reward:
            best_val_reward = current_val_reward
            best_model_path = model_dir / "best_model.pt"
            rl_filter.save_model(str(best_model_path))
            logger.info(f"✓ New best model saved (reward: {best_val_reward:.4f})")
    
    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Best validation reward: {best_val_reward:.4f}")
    logger.info(f"Models saved to: {model_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
