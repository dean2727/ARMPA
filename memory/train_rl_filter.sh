#!/bin/bash
# Train RL Memory Filter on 198 OOD Twin Tasks with Cue-based Memories
# 
# This script runs 5 epochs of GRPO training on semantically similar "twin" tasks
# that are out-of-distribution (OOD) relative to the original 200 training tasks.
#
# Key parameters:
#   - 198 twin tasks × 5 epochs = 990 cycles
#   - 1 task per cycle, 3 GRPO samples per task
#   - Dropout (0.3) enables stochastic gate probabilities for GRPO exploration
#   - Memory source: cues (step-level memories, not abstracted reasoningbank)
#
# Usage:
#   bash scripts/train_twins_cues.sh
#
# Output:
#   - Model checkpoints: models/rl_filter_twins_cues/
#   - TensorBoard logs: models/rl_filter_twins_cues/tensorboard_logs/
#   - Analysis logs: models/rl_filter_twins_cues/analysis_logs_cycle_*.jsonl

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "🚀 Starting RL Filter Training on Twins"
echo "=========================================="
echo "Tasks: 198 OOD twins"
echo "Epochs: 5 (990 total cycles)"
echo "Memory source: cues"
echo "GRPO samples per task: 3"
echo "=========================================="

python memory/train_rl_filter_online.py \
    --model_dir models/rl_filter_v6 \
    --num_cycles 990 \
    --tasks_per_cycle 1 \
    --num_samples_per_task 3 \
    --model "together_ai/OpenAI/gpt-oss-120B" \
    --instruction_path "webarena/agent/prompts/raw/p_cot_id_actree_2s_no_na_memory.py" \
    --temperature 0.7 \
    --num_memories 3 \
    --memory_source cues \
    --learning_rate 3e-4 \
    --clip_epsilon 0.2 \
    --kl_beta 0.01 \
    --gamma 0.5 \
    --disable_early_stopping \
    --fixed_task_ids "203,205,209,210,213,218,219,221,228,230,231,232,238,242,243,248,249,250,265,272,275,287,288,289,292,294,295,298,299,304,308,315,316,318,319,323,326,338,341,359,364,367,369,371,379,381,385,391,402,403,404,405,410,412,418,419,427,429,436,439,442,451,452,459,463,464,468,473,474,477,489,495,496,498,499,502,513,514,521,529,531,537,545,549,567,574,575,576,578,584,597,599,600,601,603,605,607,610,611,614,619,620,622,628,636,637,638,640,641,642,645,650,653,658,660,666,678,679,684,685,687,695,704,711,714,722,725,729,730,732,737,741,742,749,750,752,758,773,779,782,785,786,790,792,795,798,800,805,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949"

echo ""
echo "=========================================="
echo "✅ Training complete!"
echo "=========================================="
