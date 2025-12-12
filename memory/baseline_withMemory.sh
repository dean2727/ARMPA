#!/bin/bash
# Baseline: Run 148 tasks with Cue-based Memories but NO RL Filter
# 
# This script runs the same 148 tasks that were used in RL filter training v6,
# but WITHOUT the RL filter - just raw cue-based memory retrieval.
# This provides an apples-to-apples comparison to assess filter effectiveness.
#
# Comparison:
#   - Training run (v6): 148 tasks with RL filter + cues
#   - This baseline: 148 tasks with cues only (no filter)
#
# Usage:
#   bash scripts/baseline_cues_no_filter.sh
#
# Output:
#   - Results in: webarena/results_baseline_cues_no_filter/

cd "$(dirname "$0")/../webarena" || exit 1

# First 148 task IDs from the training script (matching the 148 cycles completed)
TASK_IDS="203,205,209,210,213,218,219,221,228,230,231,232,238,242,243,248,249,250,265,272,275,287,288,289,292,294,295,298,299,304,308,315,316,318,319,323,326,338,341,359,364,367,369,371,379,381,385,391,402,403,404,405,410,412,418,419,427,429,436,439,442,451,452,459,463,464,468,473,474,477,489,495,496,498,499,502,513,514,521,529,531,537,545,549,567,574,575,576,578,584,597,599,600,601,603,605,607,610,611,614,619,620,622,628,636,637,638,640,641,642,645,650,653,658,660,666,678,679,684,685,687,695,704,711,714,722,725,729,730,732,737,741,742,749,750,752,758,773,779,782,785,786,790,792,795,798,800"

# Convert comma-separated to array
IFS=',' read -ra TASK_ARRAY <<< "$TASK_IDS"
NUM_TASKS=${#TASK_ARRAY[@]}

echo "=========================================="
echo "🔬 Baseline: Cues WITHOUT RL Filter"
echo "=========================================="
echo "Tasks: $NUM_TASKS (same as training v6)"
echo "Memory source: cues"
echo "Filter: NONE (baseline)"
echo "=========================================="

RESULT_DIR="results_baseline_cues_no_filter"
mkdir -p "$RESULT_DIR"

# Run each task
for TASK_ID in "${TASK_ARRAY[@]}"; do
    echo ""
    echo ">>> Running task $TASK_ID..."
    
    python run.py \
        --agent_type litellm \
        --instruction_path "agent/prompts/raw/p_cot_id_actree_2s_no_na_memory.py" \
        --model "together_ai/OpenAI/gpt-oss-120B" \
        --temperature 0.7 \
        --get_memory \
        --num_memories 3 \
        --memory_source cues \
        --recall_threshold 0.0 \
        --test_start_idx "$TASK_ID" \
        --test_end_idx "$((TASK_ID + 1))" \
        --result_dir "$RESULT_DIR"
    
    echo "<<< Completed task $TASK_ID"
done

echo ""
echo "=========================================="
echo "✅ Baseline run complete!"
echo "=========================================="
echo "Results saved to: webarena/$RESULT_DIR"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_baseline_comparison.py"
