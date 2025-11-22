"""
Populate Cue-Based Memories from Raw Trajectory Files

This script extracts cue-based memories (observation → action mappings) from
pickled trajectory files and stores them in Qdrant for RL filter training.

Based on: webarena/trajectories_to_memories.ipynb

Usage:
    python memory/populate_cues_from_trajectories.py \
        --trajectory_dirs 20251117213558_gpt_full 20251118091613_gpt_full_2 20251118161129_qwen_full \
        --collection_name webarena

Requirements:
    - Trajectory directories must contain:
        * trajectories/ folder with .pkl files
        * results_cleaned.csv with task_id, task, success, steps columns
    - config_files/ directory with task definitions (0.json, 1.json, etc.)
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "webarena"))

from memory.manager import MemoryManager


def load_task_id_to_task_mapping(config_dir: str = "config_files", num_tasks: int = 200) -> Dict[int, str]:
    """
    Load mapping from task_id (pickle filename) to task description.
    
    Args:
        config_dir: Directory containing task config JSON files
        num_tasks: Number of tasks to load (0 to num_tasks-1)
    
    Returns:
        Dictionary mapping task_id (int) to task description (str)
    """
    task_id_to_task = {}
    
    for i in range(num_tasks):
        config_file = Path(config_dir) / f"{i}.json"
        if not config_file.exists():
            print(f"Warning: Config file {config_file} not found, skipping")
            continue
            
        with open(config_file) as f:
            config = json.load(f)
            intent = config["intent"]
            task_id = config["task_id"]
            task_id_to_task[task_id] = intent
    
    print(f"Loaded {len(task_id_to_task)} task definitions")
    return task_id_to_task


def store_trajectory_metadata(
    memory_manager: MemoryManager,
    results_df: pd.DataFrame
):
    """
    Store trajectory-level metadata (task, steps, success) for reward calculation.
    
    Args:
        memory_manager: MemoryManager instance
        results_df: DataFrame with columns: task, success, steps
    """
    print("\n📊 Storing trajectory metadata...")
    
    # Filter out any duplicate header rows
    results_df = results_df[results_df['success'] != 'success'].copy()
    
    for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Trajectory metadata"):
        success = int(row['success']) == 1
        steps = int(row['steps'])
        
        memory_manager.store_trajectory_history(
            goal=row['task'],
            num_steps=steps,
            success=success
        )
    
    print(f"✅ Stored {len(results_df)} trajectory metadata entries")


def store_cues_from_summarized_trajectories(
    memory_manager: MemoryManager,
    run_path: str,
    task_id_to_task: Dict[int, str]
):
    """
    Store cue-based memories from pre-summarized trajectory files.
    
    This is the FAST path - if trajectories were already summarized during
    the original run (with --store_memory flag), we can load them directly.
    
    Args:
        memory_manager: MemoryManager instance
        run_path: Path to run directory (e.g., "20251117213558_gpt_full")
        task_id_to_task: Mapping from task_id to task description
    """
    summarized_dir = Path(run_path) / "summarized_trajectories"
    
    if not summarized_dir.exists():
        print(f"⚠️  No summarized_trajectories/ folder found in {run_path}")
        print(f"    Skipping fast path - will need to process raw trajectories")
        return False
    
    results_csv = Path(run_path) / "results_cleaned.csv"
    if results_csv.exists():
        df_results = pd.read_csv(results_csv)
        df_results['task'] = df_results['task'].str.replace('""', '"')
        # Filter out duplicate header rows
        df_results = df_results[df_results['success'] != 'success'].copy()
    
    pickle_files = sorted(os.listdir(summarized_dir))
    
    print(f"\n📦 Processing summarized trajectories from {run_path}...")
    print(f"   Found {len(pickle_files)} pickle files")
    
    for pickle_file in tqdm(pickle_files, desc="Storing cues"):
        with open(summarized_dir / pickle_file, "rb") as f:
            observations_actions_reasonings = pickle.load(f)
        
        task_id = int(pickle_file.replace(".pkl", ""))
        task = task_id_to_task.get(task_id)
        
        if task is None:
            print(f"Warning: No task found for task_id {task_id}, skipping")
            continue
        
        # Find corresponding result row
        row = df_results[df_results['task'] == task].values
        if len(row) == 0:
            print(f"Warning: No result found for task: {task} (ID={task_id}), skipping")
            continue
        
        row = row[0]
        success = int(row[2]) == 1  # success column
        
        # Store trajectory
        memory_manager.store_trajectory(
            observations_actions_reasonings=observations_actions_reasonings,
            goal=task,
            success=success,
        )
    
    print(f"✅ Processed {len(pickle_files)} trajectories from {run_path}")
    return True


def get_already_processed_tasks(memory_manager: MemoryManager) -> set:
    """
    Get set of tasks that have already been processed and stored in Qdrant.
    
    Args:
        memory_manager: MemoryManager instance
    
    Returns:
        Set of task goals (strings) that have been processed
    """
    try:
        # Scroll through all trajectory history to find processed tasks
        points, _ = memory_manager.client.scroll(
            collection_name=memory_manager.collection_trajectory_history,
            limit=10000,  # Get all (should be ~600 max)
            with_payload=True
        )
        
        processed_goals = {p.payload['goal'] for p in points}
        return processed_goals
    except Exception as e:
        print(f"⚠️  Could not check processed tasks: {e}")
        return set()


def store_cues_from_raw_trajectories(
    memory_manager: MemoryManager,
    run_path: str,
    task_id_to_task: Dict[int, str],
    skip_existing: bool = True
):
    """
    Store cue-based memories from raw trajectory files.
    
    This is the SLOW path - we need to load and summarize each trajectory.
    Requires WebArena environment dependencies.
    
    Args:
        memory_manager: MemoryManager instance
        run_path: Path to run directory
        task_id_to_task: Mapping from task_id to task description
        skip_existing: If True, skip tasks already in Qdrant (avoid duplicates)
    """
    trajectories_dir = Path(run_path) / "trajectories"
    
    if not trajectories_dir.exists():
        print(f"❌ No trajectories/ folder found in {run_path}")
        return False
    
    results_csv = Path(run_path) / "results_cleaned.csv"
    if not results_csv.exists():
        results_csv = Path(run_path) / "results.csv"
    
    df_results = pd.read_csv(results_csv)
    df_results['task'] = df_results['task'].str.replace('""', '"')
    # Filter out duplicate header rows
    df_results = df_results[df_results['success'] != 'success'].copy()
    
    # Get already processed tasks to skip
    already_processed = get_already_processed_tasks(memory_manager) if skip_existing else set()
    if already_processed:
        print(f"📊 Found {len(already_processed)} already processed tasks - will skip them")
    
    pickle_files = sorted([f for f in os.listdir(trajectories_dir) if f.endswith('.pkl')])
    
    print(f"\n📦 Processing raw trajectories from {run_path}...")
    print(f"   Found {len(pickle_files)} pickle files")
    print(f"   ⚠️  This may take a while (need to summarize observations)...")
    
    skipped_count = 0
    processed_count = 0
    error_count = 0
    
    for pickle_file in tqdm(pickle_files, desc="Summarizing & storing"):
        try:
            with open(trajectories_dir / pickle_file, "rb") as f:
                trajectory = pickle.load(f)
        except Exception as e:
            print(f"\n⚠️  Error loading {pickle_file}: {e}")
            error_count += 1
            continue
        
        task_id = int(pickle_file.replace(".pkl", ""))
        task = task_id_to_task.get(task_id)
        
        if task is None:
            print(f"\nWarning: No task found for task_id {task_id}, skipping")
            error_count += 1
            continue
        
        # Skip if already processed
        if skip_existing and task in already_processed:
            skipped_count += 1
            continue
        
        # Find corresponding result row
        row = df_results[df_results['task'] == task].values
        if len(row) == 0:
            print(f"\nWarning: No result found for task: {task} (ID={task_id}), skipping")
            error_count += 1
            continue
        
        row = row[0]
        success = int(row[2]) == 1
        
        # Use store_trajectory_testing which handles everything internally
        # Wrap in try-except to skip problematic observations
        try:
            memory_manager.store_trajectory_testing(
                trajectory=trajectory,
                goal=task,
                success=success,
                prompt_constructor=None
            )
            processed_count += 1
        except Exception as e:
            print(f"\n⚠️  Error processing task {task_id}: {e}")
            print(f"   Skipping this trajectory and continuing...")
            error_count += 1
            continue
    
    print(f"\n✅ Summary for {run_path}:")
    print(f"   Processed: {processed_count}")
    print(f"   Skipped (already in DB): {skipped_count}")
    print(f"   Errors: {error_count}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Populate cue-based memories from trajectories")
    parser.add_argument(
        "--trajectory_dirs",
        nargs="+",
        required=True,
        help="Paths to trajectory directories (e.g., 20251117213558_gpt_full)"
    )
    parser.add_argument(
        "--collection_name",
        default="webarena",
        help="Qdrant collection name for storing cues (default: webarena)"
    )
    parser.add_argument(
        "--config_dir",
        default="webarena/config_files",
        help="Directory containing task config files (default: webarena/config_files)"
    )
    parser.add_argument(
        "--skip_metadata",
        action="store_true",
        help="Skip storing trajectory metadata (only store cues)"
    )
    parser.add_argument(
        "--force_raw",
        action="store_true",
        help="Force processing raw trajectories (skip summarized if available)"
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Don't skip already processed tasks (will create duplicates!)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("POPULATING CUE-BASED MEMORIES FROM TRAJECTORIES")
    print("="*80)
    
    # Initialize memory manager
    print(f"\n🔧 Initializing MemoryManager (collection: {args.collection_name})")
    memory_manager = MemoryManager(collection_name=args.collection_name)
    
    # Load task mappings
    print(f"\n📖 Loading task definitions from {args.config_dir}/")
    task_id_to_task = load_task_id_to_task_mapping(args.config_dir)
    
    # Process each trajectory directory
    total_stored = 0
    
    for traj_dir in args.trajectory_dirs:
        print(f"\n{'='*80}")
        print(f"Processing: {traj_dir}")
        print(f"{'='*80}")
        
        # Store trajectory metadata (for reward calculation)
        if not args.skip_metadata:
            results_csv = Path(traj_dir) / "results_cleaned.csv"
            if not results_csv.exists():
                results_csv = Path(traj_dir) / "results.csv"
            
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                df['task'] = df['task'].str.replace('""', '"')
                store_trajectory_metadata(memory_manager, df)
            else:
                print(f"⚠️  No results CSV found in {traj_dir}, skipping metadata")
        
        # Try fast path first (pre-summarized)
        if not args.force_raw:
            success = store_cues_from_summarized_trajectories(
                memory_manager, traj_dir, task_id_to_task
            )
            if success:
                continue
        
        # Fall back to slow path (raw trajectories)
        store_cues_from_raw_trajectories(
            memory_manager, 
            traj_dir, 
            task_id_to_task,
            skip_existing=not args.no_skip_existing
        )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Get collection info
    try:
        collection_info = memory_manager.client.get_collection(
            f"{args.collection_name}-cues"
        )
        cue_count = collection_info.points_count
        
        collection_info_hist = memory_manager.client.get_collection(
            f"{args.collection_name}-trajectory-history"
        )
        hist_count = collection_info_hist.points_count
        
        print(f"✅ Cue-based memories: {cue_count}")
        print(f"✅ Trajectory metadata: {hist_count}")
        print(f"\n🎉 Ready for RL training!")
        
    except Exception as e:
        print(f"⚠️  Could not retrieve collection stats: {e}")
    
    print("="*80)


if __name__ == '__main__':
    main()
