# RL Memory Filter Agent

Reinforcement Learning agent that learns to select useful memories for WebArena web navigation tasks.

## Overview

The RL Memory Filter Agent scores retrieved memories based on their predicted usefulness for task completion. It uses:
- **Neural Network Policy**: Scores each memory with continuous values [0, 1]
- **PPO Training**: Learns from task success/failure signals
- **Behavioral Cloning Pre-training**: Warm-start from successful trajectories
- **Threshold-based Selection**: Selects memories with score ≥ threshold (default 0.6)

## Architecture

```
State: [memory_embeddings (k × 384), task_embedding (384), obs_embedding (384), entropy (1)]
       ↓
Policy Network: 
  - Context encoder (task + observation + entropy)
  - Memory encoder (per-memory features)
  - Scoring head (combines context + memory → score)
       ↓
Action: k continuous scores in [0, 1]
       ↓
Selection: Keep memories with score ≥ threshold
```

## Implementation Files

### Core Components

1. **`memory/rl_filter_agent.py`**
   - `MemoryFilterPolicy`: Neural network for scoring memories
   - `MemoryFilterEnv`: Gymnasium environment (for future online RL)
   - `RLMemoryFilter`: Inference wrapper for production use

2. **`train_rl_filter.py`**
   - PPO training script
   - TensorBoard logging
   - Checkpoint management
   - Evaluation callbacks

3. **`webarena/run.py`** (modified)
   - Added `--collect_rl_data`: Logs memory states for training
   - Added `--use_rl_filter`: Uses trained model for inference
   - Data collection logic at each step
   - Episode data saving

4. **`RL_IMPLEMENTATION_PLAN.md`**
   - Complete implementation strategy
   - Design decisions and rationale
   - Training timeline

## Usage

### Step 1: Collect Training Data

Run WebArena tasks with data collection enabled:

```bash
cd webarena
export PYTHONPATH="/path/to/ARMPA:/path/to/ARMPA/webarena:$PYTHONPATH"

# Collect data from 50 training tasks
.venv/bin/python run.py \
  --instruction_path agent/prompts/raw/p_cot_id_actree_2s_no_na_memory.py \
  --agent_type litellm \
  --model together_ai/Qwen/Qwen2.5-72B-Instruct \
  --temperature 0.7 \
  --test_start_idx 0 \
  --test_end_idx 50 \
  --get_memory \
  --store_memory \
  --collect_rl_data \
  --num_memories 10 \
  --max_steps 30
```

This saves training data to `runs/<timestamp>/rl_training_data/task_*.pkl`

**Expected output:**
- ~50 episode files
- Each contains: memory states, task context, actions, final success
- 3-5 hours on Mac M-series (depending on task complexity)

### Step 2: Behavioral Cloning Pre-training

Train supervised model on successful trajectories:

```bash
# TODO: Create bc_pretrain.py script
.venv/bin/python bc_pretrain.py \
  --data_dir runs/<timestamp>/rl_training_data \
  --output_dir bc_models \
  --epochs 50 \
  --batch_size 32
```

This creates initial policy by imitating successful memory selections.

### Step 3: PPO Fine-tuning

Fine-tune with reinforcement learning:

```bash
.venv/bin/python train_rl_filter.py \
  --train_start 0 \
  --train_end 100 \
  --val_start 100 \
  --val_end 120 \
  --pretrained_model bc_models/best_model.pt \
  --total_timesteps 100000 \
  --device mps
```

Monitor training:
```bash
tensorboard --logdir rl_training_runs/rl_filter_*/logs
```

**Expected training time:** 1-2 hours on Mac M-series

### Step 4: Inference (Use Trained Model)

Run WebArena with RL-filtered memories:

```bash
.venv/bin/python run.py \
  --instruction_path agent/prompts/raw/p_cot_id_actree_2s_no_na_memory.py \
  --agent_type litellm \
  --model together_ai/Qwen/Qwen2.5-72B-Instruct \
  --temperature 0.7 \
  --test_start_idx 100 \
  --test_end_idx 130 \
  --get_memory \
  --use_rl_filter \
  --rl_filter_model rl_training_runs/rl_filter_*/best_model/best_model.zip \
  --rl_filter_threshold 0.6
```

### Step 5: Evaluation

Compare different strategies:

```bash
# Baseline: No memory
.venv/bin/python run.py ... --test_start_idx 100 --test_end_idx 130

# All memories
.venv/bin/python run.py ... --test_start_idx 100 --test_end_idx 130 --get_memory

# RL-filtered memories
.venv/bin/python run.py ... --test_start_idx 100 --test_end_idx 130 --get_memory --use_rl_filter --rl_filter_model <model_path>
```

Compare success rates, average steps, memory efficiency.

## Training Data Format

Each episode file (`task_*.pkl`) contains:

```python
{
    'task_id': int,
    'config_file': str,
    'intent': str,
    'success': bool,  # 1.0 or 0.0
    'num_steps': int,
    'steps': [
        {
            'memory_embeddings': List[np.array],  # (k, 384)
            'task_embedding': np.array,            # (384,)
            'obs_embedding': np.array,             # (384,)
            'entropy': float,
            'action_decision_entropy': float,
            'num_memories': int,
            'memories_used': List[int],  # Indices
        },
        ...
    ]
}
```

## Model Checkpoints

Trained models are saved in:
```
rl_training_runs/rl_filter_<timestamp>/
├── best_model/           # Best model on validation set
│   └── best_model.zip
├── checkpoints/          # Periodic checkpoints
│   ├── rl_filter_10000_steps.zip
│   ├── rl_filter_20000_steps.zip
│   └── ...
├── logs/                 # TensorBoard logs
│   ├── PPO_1/
│   └── eval/
├── config.json          # Training configuration
└── training_summary.json
```

## Design Decisions

### Why Offline RL + PPO?

- **Data Efficiency**: WebArena tasks expensive (LLM + browser)
- **Reusability**: Collect once, train multiple times
- **Stability**: BC warm-start avoids random exploration
- **Sample Efficient**: PPO proven for sequential decision-making

### Why Continuous Scores?

- **Flexibility**: Adjust threshold without retraining
- **Ranking**: Natural ordering of memories
- **Differentiation**: More informative than binary yes/no
- **Gradient Flow**: Better for neural network training

### Why Task Success Reward?

- **Simple**: Already available from WebArena
- **Aligned**: True objective is task completion
- **No overhead**: No additional LLM calls
- **Proven**: Standard sparse reward in RL

Future: Can add LLM-judge for dense rewards if needed.

## Configuration

### Key Hyperparameters

```python
# Memory Filter
max_memories = 10          # Max memories to retrieve
score_threshold = 0.6      # Min score to include memory
embedding_dim = 384        # Embedding dimension (from MemoryManager)

# PPO Training
learning_rate = 3e-4       # Adam learning rate
n_steps = 2048             # Steps per rollout
batch_size = 64            # Mini-batch size
n_epochs = 10              # Optimization epochs per update
gamma = 0.99               # Discount factor
gae_lambda = 0.95          # GAE parameter
clip_range = 0.2           # PPO clip range
ent_coef = 0.01            # Entropy bonus
```

### Tuning Recommendations

- **Increase `score_threshold`**: More selective, fewer memories
- **Decrease `score_threshold`**: Less selective, more memories
- **Increase `n_steps`**: Better gradient estimates, slower training
- **Increase `ent_coef`**: More exploration during training

## Troubleshooting

### Issue: "RL Filter Agent not available"
**Solution**: Check that `memory/rl_filter_agent.py` exists and imports successfully:
```bash
.venv/bin/python -c "from memory.rl_filter_agent import RLMemoryFilter; print('OK')"
```

### Issue: "Model checkpoint not found"
**Solution**: Verify model path exists:
```bash
ls -la rl_training_runs/rl_filter_*/best_model/best_model.zip
```

### Issue: No memories returned after filtering
**Solution**: Lower threshold or check that memories have non-zero scores:
```bash
--rl_filter_threshold 0.4  # Lower threshold
```

### Issue: MPS not available
**Solution**: Falls back to CPU automatically. Check PyTorch installation:
```bash
.venv/bin/python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

## Performance Expectations

Based on preliminary design:

**Training**:
- Data collection: 3-5 hours (50 tasks)
- BC pre-training: 30 mins
- PPO fine-tuning: 1-2 hours
- **Total: 5-8 hours**

**Inference**:
- Overhead: <100ms per step (memory scoring)
- Negligible compared to LLM inference (~2-5 seconds)

**Expected Improvements**:
- Success rate: +5-15% over fixed memory selection
- Efficiency: 20-30% fewer memories used on average
- Adaptability: Better handling of high/low entropy situations

## Next Steps

1. ✅ Core RL components implemented
2. ✅ Data collection integrated into run.py
3. ⏳ Collect training data (50-100 tasks)
4. ⏳ Implement BC pre-training script
5. ⏳ Train and validate model
6. ⏳ Run ablation studies
7. ⏳ Document final results

## References

- **PPO Paper**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **Stable-Baselines3**: [Documentation](https://stable-baselines3.readthedocs.io/)
- **WebArena**: [Original benchmark](https://webarena.dev/)

## Contact

For questions or issues, refer to the implementation plan in `RL_IMPLEMENTATION_PLAN.md`.

---

**Author**: ARMPA Team  
**Date**: November 2025  
**Status**: Core implementation complete, ready for data collection
