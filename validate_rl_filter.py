"""
Validation Study: Ablation Comparison

Compares 3 conditions:
1. No memory baseline
2. All memories (no filter)
3. RL-filtered memories

Runs each condition on held-out test tasks and compares performance.

Usage:
    python validate_rl_filter.py --rl_model models/ppo_trained/best_model \
                                   --num_test_tasks 30 \
                                   --output_dir validation_results
"""

import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import subprocess
import json


def run_condition(condition_name, num_tasks, max_steps, num_products, extra_args=''):
    """Run WebShop tasks under specific condition"""
    
    print(f"\n{'='*80}")
    print(f"Running: {condition_name}")
    print(f"{'='*80}")
    
    cmd = [
        'python', 'run_webshop.py',
        '--num_tasks', str(num_tasks),
        '--max_steps', str(max_steps),
        '--num_products', str(num_products),
    ]
    
    if extra_args:
        cmd.extend(extra_args.split())
    
    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error running {condition_name}:")
        print(result.stderr)
        return None
    
    # Find the results file from output
    lines = result.stdout.split('\n')
    results_file = None
    for line in lines:
        if 'Results saved to:' in line:
            results_file = line.split('Results saved to:')[1].strip()
            break
    
    if not results_file:
        print(f"❌ Could not find results file for {condition_name}")
        return None
    
    # Load results
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    return results


def compute_metrics(results):
    """Compute performance metrics from results"""
    
    if not results:
        return None
    
    return {
        'success_rate': np.mean([r['success'] for r in results]),
        'avg_reward': np.mean([r['reward'] for r in results]),
        'avg_steps': np.mean([r['steps'] for r in results]),
        'std_reward': np.std([r['reward'] for r in results]),
        'num_tasks': len(results),
    }


def run_ablation_study(num_tasks, max_steps, num_products, rl_model_path=None):
    """Run all three conditions"""
    
    results = {}
    
    # Condition 1: No memory baseline
    print("\n🔬 CONDITION 1: No Memory Baseline")
    results['no_memory'] = run_condition(
        'No Memory',
        num_tasks=num_tasks,
        max_steps=max_steps,
        num_products=num_products,
        extra_args='',
    )
    
    # Condition 2: All memories (no filter)
    print("\n🔬 CONDITION 2: All Memories (No Filter)")
    results['all_memories'] = run_condition(
        'All Memories',
        num_tasks=num_tasks,
        max_steps=max_steps,
        num_products=num_products,
        extra_args='--get_memory --num_memories 5',
    )
    
    # Condition 3: RL-filtered memories
    if rl_model_path:
        print("\n🔬 CONDITION 3: RL-Filtered Memories")
        results['rl_filtered'] = run_condition(
            'RL Filtered',
            num_tasks=num_tasks,
            max_steps=max_steps,
            num_products=num_products,
            extra_args=f'--get_memory --use_rl_filter --rl_model_path {rl_model_path} --num_memories 5',
        )
    else:
        print("\n⚠️  Skipping RL-filtered condition (no model provided)")
        results['rl_filtered'] = None
    
    return results


def analyze_results(results, output_dir):
    """Statistical analysis and visualization"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    metrics = {}
    for condition, data in results.items():
        if data is not None:
            metrics[condition] = compute_metrics(data)
    
    # Create comparison table
    df = pd.DataFrame(metrics).T
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(df.to_string())
    print()
    
    # Save metrics
    df.to_csv(output_dir / 'metrics_comparison.csv')
    
    # Statistical tests
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS (t-tests)")
    print("="*80)
    
    comparisons = []
    
    if results['no_memory'] and results['all_memories']:
        rewards_no_mem = [r['reward'] for r in results['no_memory']]
        rewards_all_mem = [r['reward'] for r in results['all_memories']]
        t_stat, p_value = stats.ttest_ind(rewards_no_mem, rewards_all_mem)
        comparisons.append({
            'comparison': 'No Memory vs All Memories',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        })
        print(f"No Memory vs All Memories: t={t_stat:.3f}, p={p_value:.4f} {'✓ Significant' if p_value < 0.05 else '✗ Not significant'}")
    
    if results['all_memories'] and results['rl_filtered']:
        rewards_all_mem = [r['reward'] for r in results['all_memories']]
        rewards_rl = [r['reward'] for r in results['rl_filtered']]
        t_stat, p_value = stats.ttest_ind(rewards_all_mem, rewards_rl)
        comparisons.append({
            'comparison': 'All Memories vs RL Filtered',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        })
        print(f"All Memories vs RL Filtered: t={t_stat:.3f}, p={p_value:.4f} {'✓ Significant' if p_value < 0.05 else '✗ Not significant'}")
    
    if results['no_memory'] and results['rl_filtered']:
        rewards_no_mem = [r['reward'] for r in results['no_memory']]
        rewards_rl = [r['reward'] for r in results['rl_filtered']]
        t_stat, p_value = stats.ttest_ind(rewards_no_mem, rewards_rl)
        comparisons.append({
            'comparison': 'No Memory vs RL Filtered',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        })
        print(f"No Memory vs RL Filtered: t={t_stat:.3f}, p={p_value:.4f} {'✓ Significant' if p_value < 0.05 else '✗ Not significant'}")
    
    # Save statistical tests
    pd.DataFrame(comparisons).to_csv(output_dir / 'statistical_tests.csv', index=False)
    
    # Visualizations
    print(f"\nGenerating visualizations...")
    
    # 1. Bar plot: Average rewards
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = [k for k, v in metrics.items() if v is not None]
    rewards = [metrics[c]['avg_reward'] for c in conditions]
    errors = [metrics[c]['std_reward'] for c in conditions]
    
    bars = ax.bar(conditions, rewards, yerr=errors, capsize=5, alpha=0.7)
    ax.set_ylabel('Average Reward')
    ax.set_title('Performance Comparison: Average Reward by Condition')
    ax.grid(axis='y', alpha=0.3)
    
    # Color bars
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_comparison.png', dpi=150)
    print(f"✓ Saved reward_comparison.png")
    
    # 2. Success rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    success_rates = [metrics[c]['success_rate'] * 100 for c in conditions]
    
    bars = ax.bar(conditions, success_rates, alpha=0.7)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Performance Comparison: Success Rate by Condition')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_comparison.png', dpi=150)
    print(f"✓ Saved success_rate_comparison.png")
    
    # 3. Box plot: Reward distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    reward_data = []
    labels = []
    
    for condition in conditions:
        if results[condition]:
            reward_data.append([r['reward'] for r in results[condition]])
            labels.append(condition.replace('_', ' ').title())
    
    bp = ax.boxplot(reward_data, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Reward')
    ax.set_title('Reward Distribution by Condition')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_distribution.png', dpi=150)
    print(f"✓ Saved reward_distribution.png")
    
    print(f"\n✓ All results saved to {output_dir}")
    
    return metrics, comparisons


def generate_report(metrics, comparisons, output_dir):
    """Generate markdown report"""
    
    report = f"""# RL Memory Filter Validation Report

## Experimental Setup

- **Test Tasks**: {list(metrics.values())[0]['num_tasks']} per condition
- **Max Steps**: 15 per episode
- **Products**: 1000 WebShop products

## Results Summary

### Performance Metrics

| Condition | Success Rate | Avg Reward | Avg Steps |
|-----------|-------------|-----------|-----------|
"""
    
    for condition, m in metrics.items():
        condition_name = condition.replace('_', ' ').title()
        report += f"| {condition_name} | {m['success_rate']*100:.1f}% | {m['avg_reward']:.3f} | {m['avg_steps']:.1f} |\n"
    
    report += f"""
### Statistical Significance

"""
    
    for comp in comparisons:
        sig = "✓ Significant (p < 0.05)" if comp['significant'] else "✗ Not significant"
        report += f"- **{comp['comparison']}**: t={comp['t_statistic']:.3f}, p={comp['p_value']:.4f} - {sig}\n"
    
    report += """

## Analysis

### Key Findings

"""
    
    # Determine if RL filter helped
    if 'rl_filtered' in metrics and 'all_memories' in metrics:
        rl_reward = metrics['rl_filtered']['avg_reward']
        all_reward = metrics['all_memories']['avg_reward']
        improvement = ((rl_reward - all_reward) / abs(all_reward + 1e-10)) * 100
        
        if improvement > 0:
            report += f"1. **RL filter improved performance**: {improvement:.1f}% increase in average reward compared to using all memories\n"
        else:
            report += f"1. **RL filter did not improve performance**: {abs(improvement):.1f}% decrease in average reward compared to using all memories\n"
    
    if 'no_memory' in metrics and 'all_memories' in metrics:
        all_reward = metrics['all_memories']['avg_reward']
        no_reward = metrics['no_memory']['avg_reward']
        improvement = ((all_reward - no_reward) / abs(no_reward + 1e-10)) * 100
        
        if improvement > 0:
            report += f"2. **Memory retrieval is beneficial**: {improvement:.1f}% increase compared to no memory baseline\n"
        else:
            report += f"2. **Memory retrieval not beneficial**: {abs(improvement):.1f}% decrease compared to no memory baseline\n"
    
    report += """

### Interpretation

The results demonstrate whether learning to filter memories provides value over:
- Not using memories at all (baseline)
- Using all retrieved memories without filtering

Statistical significance tests indicate whether observed differences are likely due to chance.

## Visualizations

See accompanying PNG files:
- `reward_comparison.png`: Bar chart of average rewards
- `success_rate_comparison.png`: Success rates by condition
- `reward_distribution.png`: Box plots showing reward distributions

"""
    
    # Save report
    with open(Path(output_dir) / 'validation_report.md', 'w') as f:
        f.write(report)
    
    print(f"\n✓ Generated validation_report.md")


def main():
    parser = argparse.ArgumentParser(description="Validation Ablation Study")
    parser.add_argument('--rl_model', type=str, default=None,
                        help='Path to trained RL model (optional)')
    parser.add_argument('--num_test_tasks', type=int, default=30,
                        help='Number of test tasks per condition')
    parser.add_argument('--max_steps', type=int, default=15,
                        help='Max steps per task')
    parser.add_argument('--num_products', type=int, default=1000,
                        help='Number of products in WebShop')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RL MEMORY FILTER VALIDATION STUDY")
    print("="*80)
    
    # Run ablation study
    results = run_ablation_study(
        num_tasks=args.num_test_tasks,
        max_steps=args.max_steps,
        num_products=args.num_products,
        rl_model_path=args.rl_model,
    )
    
    # Analyze results
    metrics, comparisons = analyze_results(results, args.output_dir)
    
    # Generate report
    generate_report(metrics, comparisons, args.output_dir)
    
    print("\n" + "="*80)
    print("✓ VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
