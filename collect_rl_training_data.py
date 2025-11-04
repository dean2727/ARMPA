"""
Data Collection Script for RL Memory Filter Training

This script collects training data by running WebArena tasks with different
memory configurations. Each episode records:
- Retrieved memories and their embeddings
- Task/observation context
- Agent entropy
- Memory selection (all, random subset, etc.)
- Final task success

This data is then used for offline RL training.

Usage:
    python collect_rl_training_data.py --tasks 0-50 --output data/rl_training

Author: ARMPA Team
Date: November 2025
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# This will run WebArena tasks and collect memory selection data
# We'll save: (state, action, reward) tuples where:
# - state = (memory_embeddings, task_emb, obs_emb, entropy)
# - action = which memories were used (binary mask or scores)
# - reward = task success (1.0 or 0.0)


def collect_episode_data(
    task_id: int,
    config_file: str,
    memory_selection_strategy: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Run one WebArena episode and collect memory selection data.
    
    Args:
        task_id: Task identifier
        config_file: Path to task config
        memory_selection_strategy: Strategy for selecting memories
            - 'all': Use all retrieved memories
            - 'random_k': Use random k memories
            - 'top_k': Use top k by score
        args: Command-line arguments
    
    Returns:
        episode_data: Dictionary containing trajectory data
    """
    # TODO: This needs to integrate with webarena/run.py
    # For now, return placeholder structure
    
    episode_data = {
        'task_id': task_id,
        'config_file': config_file,
        'strategy': memory_selection_strategy,
        'steps': [],
        'success': False,
        'total_steps': 0,
    }
    
    # Each step would contain:
    # step_data = {
    #     'memory_embeddings': np.array,  # (num_memories, 384)
    #     'task_embedding': np.array,      # (384,)
    #     'obs_embedding': np.array,       # (384,)
    #     'entropy': float,
    #     'memories_used': List[int],      # Indices of memories used
    #     'num_memories_available': int,
    # }
    
    return episode_data


def main(args: argparse.Namespace):
    """
    Main data collection function.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"collection_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Collecting RL Training Data")
    print(f"{'='*80}")
    print(f"Tasks: {args.task_start}-{args.task_end}")
    print(f"Output: {run_dir}")
    print(f"{'='*80}\n")
    
    # Save collection config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Collect data for each task
    all_episodes = []
    
    for task_id in tqdm(range(args.task_start, args.task_end), desc="Collecting episodes"):
        config_file = f"webarena/config_files/{task_id}.json"
        
        if not Path(config_file).exists():
            continue
        
        # Collect episodes with different memory strategies
        for strategy in args.strategies:
            episode = collect_episode_data(
                task_id=task_id,
                config_file=config_file,
                memory_selection_strategy=strategy,
                args=args,
            )
            all_episodes.append(episode)
    
    # Save collected data
    data_file = run_dir / "episodes.pkl"
    with open(data_file, "wb") as f:
        pickle.dump(all_episodes, f)
    
    print(f"\nCollected {len(all_episodes)} episodes")
    print(f"Data saved to: {data_file}")
    
    # Save summary statistics
    successful = sum(1 for ep in all_episodes if ep['success'])
    summary = {
        'total_episodes': len(all_episodes),
        'successful_episodes': successful,
        'success_rate': successful / len(all_episodes) if all_episodes else 0.0,
        'data_file': str(data_file),
    }
    
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RL training data")
    
    parser.add_argument("--task_start", type=int, default=0)
    parser.add_argument("--task_end", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="data/rl_training")
    parser.add_argument("--strategies", nargs="+",
                       default=["all", "random_3", "random_5", "top_3", "top_5"])
    parser.add_argument("--max_steps", type=int, default=30)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
