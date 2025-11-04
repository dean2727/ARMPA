"""
RL Memory Filter Agent - Implementation Plan

This document outlines the complete implementation strategy for training
an RL agent to filter memories in WebArena tasks.

## Problem Statement

Given k retrieved memories for a web navigation step, the RL agent must:
- Score each memory's predicted usefulness (continuous [0,1])
- Apply threshold to select subset
- Maximize task success rate

## Implementation Strategy

### Phase 1: Data Collection (CURRENT PHASE)

Since WebArena tasks are expensive (LLM calls + browser automation), we use
an **offline RL approach** with data collection:

1. **Run WebArena tasks with memory retrieval enabled**
   - Modify run.py to log memory data at each step
   - Log: memory_embeddings, task_emb, obs_emb, entropy, final_success

2. **Vary memory selection strategies during collection**
   - All memories (baseline)
   - Random k memories
   - Top-k by retrieval score
   - Random subsets of varying sizes

3. **Collect ~100-200 task episodes**
   - Mix of successful and failed trajectories
   - Diverse tasks from WebArena benchmark

### Phase 2: Offline RL Training

**Approach: Behavioral Cloning + PPO Fine-tuning**

Step 1: **Behavioral Cloning (Supervised Pre-training)**
- Label memories in successful trajectories as "useful"
- Train policy to imitate successful memory selections
- Loss: BCE between predicted scores and success-based labels
- Fast convergence, good initialization

Step 2: **PPO Fine-tuning**
- Use pre-trained policy as initialization  
- Fine-tune with sparse reward (task success)
- Explore better memory selection strategies
- Handle cases not covered in BC data

**Why this hybrid approach?**
- BC provides strong initial policy (avoids random exploration)
- PPO refines policy based on true objective (task success)
- More sample-efficient than pure RL
- Proven effective for offline→online transfer

### Phase 3: Integration & Deployment

Minimal changes to run.py:
```python
# Add at top
from memory.rl_filter_agent import RLMemoryFilter

# Initialize filter (in test function)
if args.use_rl_filter:
    rl_filter = RLMemoryFilter(
        model_path=args.rl_filter_model,
        score_threshold=args.rl_filter_threshold,
    )

# Filter memories (in main loop)
if args.use_rl_filter:
    filtered_memories = rl_filter.filter_memories(
        memories=memories,
        task_embedding=task_emb,
        obs_embedding=obs_emb,
        entropy=current_entropy,
    )
    memories = filtered_memories
```

### Phase 4: Validation & Ablation

Compare performance:
1. **Baseline**: No memory retrieval
2. **All memories**: Use all k retrieved memories  
3. **Random selection**: Random subset
4. **RL-filtered**: Our trained agent

Metrics:
- Task success rate
- Average steps to completion
- Memory efficiency (memories used per task)

## Implementation Files

### Core Components (DONE ✓)

1. **memory/rl_filter_agent.py**
   - MemoryFilterPolicy: Neural network for scoring
   - MemoryFilterEnv: Gym environment (for online RL if needed)
   - RLMemoryFilter: Inference wrapper

2. **train_rl_filter.py**
   - Training script with PPO
   - TensorBoard logging
   - Checkpoint management

3. **collect_rl_training_data.py**
   - Data collection from WebArena runs
   - Episode logging

### Next Steps (TODO)

4. **Integrate data logging into run.py**
   - Add --collect_rl_data flag
   - Log memory states at each step
   - Save episode data for training

5. **Behavioral cloning pre-training**
   - Train on successful trajectories
   - Supervised learning with BCE loss

6. **PPO fine-tuning**
   - Load BC model
   - Fine-tune with task success reward

7. **Evaluation script**
   - Run ablation studies
   - Generate comparison metrics

## Training Timeline (Mac MPS)

Estimated times:
- Data collection: 3-5 hours (100 tasks × 2min avg)
- BC pre-training: 30 mins
- PPO fine-tuning: 1-2 hours
- Validation: 1 hour (30 tasks × 2min)
- **Total: ~6-8 hours**

## Decision Log

**Q: Why offline RL instead of online?**
A: WebArena tasks are expensive. Offline RL lets us:
   - Reuse data from multiple runs
   - Train without running new tasks
   - Faster iteration during development

**Q: Why BC + PPO instead of pure PPO?**
A: BC provides warm start, avoiding random exploration.
   PPO then refines for optimal performance.

**Q: Why not use LLM-judge rewards?**
A: Not yet implemented. Task success is simpler, proven signal.
   Can add LLM-judge later as dense reward shaping.

**Q: Can we train end-to-end online?**
A: Yes, but much slower. MemoryFilterEnv is designed for this.
   Start with offline, upgrade to online if needed.

## File Structure

```
ARMPA/
├── memory/
│   ├── rl_filter_agent.py          # Core RL components
│   ├── manager.py                   # Memory retrieval (existing)
│   └── prompts/                     # Prompts (existing)
├── webarena/
│   ├── run.py                       # Main script (modify for logging)
│   └── config_files/                # Task configs
├── train_rl_filter.py               # Training script
├── collect_rl_training_data.py      # Data collection
├── evaluate_rl_filter.py            # Evaluation (TODO)
├── rl_training_runs/                # Training outputs (created)
│   └── rl_filter_TIMESTAMP/
│       ├── checkpoints/
│       ├── logs/
│       ├── best_model/
│       └── config.json
└── data/
    └── rl_training/                 # Collected data (created)
        └── episodes.pkl
```

## Next Immediate Action

Modify run.py to add data collection mode:
1. Add --collect_rl_data flag
2. Log memory states + actions + rewards
3. Save to pickle files for offline training

Then run data collection on 50-100 tasks.

Author: ARMPA Team
Date: November 2025
"""

if __name__ == "__main__":
    print(__doc__)
