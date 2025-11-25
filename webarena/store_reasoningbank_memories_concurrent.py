#!/usr/bin/env python3
"""
Concurrently execute store_reasoningbank_memories_from_run for multiple run directories.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

# Load environment variables
load_dotenv(override=True)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from memory.manager import MemoryManager
from memory.prompts.lessons import success_prompt, failure_prompt
from webarena.llms.providers.litellm_utils import generate_from_litellm_completion


def get_formatted_trajectory_for_prompt(observations_actions_reasonings: list) -> str:
    """The ReasoningBank paper constructs a trajectory (for trajectory lesson discovery)
    by considering the per-step agent actions, its reasoning behind them,
    and the resulting (i+1) observation (summarized).
    """

    # ---- Step 0 values ----
    initial_obs = observations_actions_reasonings[0][0].strip()
    initial_action = observations_actions_reasonings[0][1].replace('\n', '').strip()
    initial_reasoning = observations_actions_reasonings[0][2].replace('\n', '').strip()

    formatted_traj = (
        f"\nINITIAL OBSERVATION:\n{initial_obs}\n"
        f"ACTION:\n{initial_action}\n"
        f"REASONING:\n{initial_reasoning}\n"
    )

    # ---- Remaining steps ----
    for i in range(1, len(observations_actions_reasonings)):
        obs_i = observations_actions_reasonings[i][0].strip()
        action_i = observations_actions_reasonings[i][1].replace('\n', '').strip()
        reasoning_i = (
            observations_actions_reasonings[i][2].replace('\n', '').strip()
            if observations_actions_reasonings[i][2]
            else None
        )

        formatted_traj += f"OUTCOME:\n{obs_i}\n\n"
        formatted_traj += f"ACTION:\n{action_i}\n"

        if reasoning_i:
            formatted_traj += f"REASONING:\n{reasoning_i}\n"

    formatted_traj += "<END OF TRAJECTORY>"

    return formatted_traj


def load_task_id_to_task_mapping():
    """Load the task_id to task mapping from config files."""
    task_id_to_task = {}
    
    st_idx = 0
    ed_idx = 200  # However many webarena tasks we will use
    test_file_list = []
    for i in range(st_idx, ed_idx):
        test_file_list.append(f"config_files/{i}.json")
    
    base_path = Path(__file__).parent
    for config_file in test_file_list:
        config_path = base_path / config_file
        if not config_path.exists():
            continue
        with open(config_path) as f:
            _c = json.load(f)
            intent = _c["intent"]
            task_id = _c["task_id"]
            task_id_to_task[task_id] = intent
    
    return task_id_to_task


def store_reasoningbank_memories_from_run(run_path: str, task_id_to_task: dict, memory_manager: MemoryManager):
    """Store ReasoningBank memories from a run directory."""
    failed_prompts = []
    base_path = Path(__file__).parent
    
    # Load the results csv
    results_path = base_path / f"{run_path}/results_cleaned.csv"
    if not results_path.exists():
        print(f"Warning: {results_path} does not exist, skipping {run_path}")
        return failed_prompts
    
    df_results = pd.read_csv(results_path)
    df_results['task'] = df_results['task'].str.replace('""', '"')
    
    summarized_trajectories_path = base_path / f"{run_path}/summarized_trajectories"
    if not summarized_trajectories_path.exists():
        print(f"Warning: {summarized_trajectories_path} does not exist, skipping {run_path}")
        return failed_prompts
    
    pickle_files = [f for f in os.listdir(summarized_trajectories_path) if f.endswith('.pkl')]
    
    print(f"Processing {len(pickle_files)} trajectories from {run_path}...")
    
    for pickle_file in tqdm(pickle_files, desc=f"Processing {run_path}"):
        try:
            pickle_path = summarized_trajectories_path / pickle_file
            observations_actions_reasonings = pickle.load(open(pickle_path, "rb"))
            formatted_traj = get_formatted_trajectory_for_prompt(observations_actions_reasonings)

            task_id = int(pickle_file.replace(".pkl", ""))
            if task_id not in task_id_to_task:
                print(f"Warning: task_id {task_id} not found in mapping, skipping")
                continue
            
            task = task_id_to_task[task_id]

            user_prompt = f"""
QUERY: {task}
TRAJECTORY:
{formatted_traj}
            """
            
            row = df_results[df_results['task_id'] == task_id].values
            if len(row) == 0:
                continue

            row = row[0]
            goal, success = row[1], row[2]

            success = int(success) == 1
            prompt = ""
            if success:
                prompt = success_prompt
            else:
                prompt = failure_prompt

            try:
                response = generate_from_litellm_completion(
                    prompt=prompt,
                    model="together_ai/OpenAI/gpt-oss-120B",
                    temperature=1.0,  # Same temperature as the original paper for memory extraction
                    max_tokens=1024,
                    system_prompt=user_prompt,
                    stop_sequences=None,
                )
                mem_items = json.loads(response['answer'])
                mem_items = mem_items['memory_items']

                memory_manager.store_reasoningbank_memories(
                    memory_items=mem_items,
                    goal=task,
                    success=success,
                    source_trajectory_id=task_id
                )
            except Exception as e:
                failed_prompts.append(task_id)
                print(f"Failed on task ID {task_id} in {run_path} - {e}")
        except Exception as e:
            print(f"Error processing {pickle_file} in {run_path}: {e}")
            continue
    
    return failed_prompts


def main():
    """Main function to run concurrent execution."""
    # Initialize memory manager
    MEMORY_COLLECTION_NAME = "webarena"
    memory_manager = MemoryManager(collection_name=MEMORY_COLLECTION_NAME)
    
    # Load task_id to task mapping
    print("Loading task_id to task mapping...")
    task_id_to_task = load_task_id_to_task_mapping()
    print(f"Loaded {len(task_id_to_task)} task mappings")
    
    # Define run paths
    run_paths = [
        "runs/20251117213558_gpt_full",
        "runs/20251118091613_gpt_full_2",
        "runs/20251118161129_qwen_full"
    ]
    
    # Execute concurrently using ThreadPoolExecutor
    print(f"\nStarting concurrent processing of {len(run_paths)} runs...")
    all_failed_prompts = {}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_run = {
            executor.submit(
                store_reasoningbank_memories_from_run,
                run_path,
                task_id_to_task,
                memory_manager
            ): run_path
            for run_path in run_paths
        }
        
        # Process completed tasks
        for future in as_completed(future_to_run):
            run_path = future_to_run[future]
            try:
                failed_prompts = future.result()
                all_failed_prompts[run_path] = failed_prompts
                print(f"\n✓ Completed {run_path}")
                if failed_prompts:
                    print(f"  Failed prompts: {len(failed_prompts)}")
            except Exception as e:
                print(f"\n✗ Error processing {run_path}: {e}")
                all_failed_prompts[run_path] = []
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    total_failed = sum(len(failed) for failed in all_failed_prompts.values())
    if total_failed > 0:
        print(f"Total failed prompts: {total_failed}")
        for run_path, failed in all_failed_prompts.items():
            if failed:
                print(f"  {run_path}: {len(failed)} failed")
    else:
        print("All prompts processed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

