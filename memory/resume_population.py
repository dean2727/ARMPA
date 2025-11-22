"""
Resume population script - continues from where it left off.

Usage:
    python memory/resume_population.py
"""

from memory.manager import MemoryManager

# Check what's already stored
mm = MemoryManager(collection_name='webarena')

try:
    info_cues = mm.client.get_collection('webarena-cues')
    info_hist = mm.client.get_collection('webarena-trajectory-history')
    
    print("="*80)
    print("CURRENT PROGRESS")
    print("="*80)
    print(f"✅ Cue-based memories stored: {info_cues.points_count}")
    print(f"✅ Trajectory metadata stored: {info_hist.points_count}")
    print()
    
    # Get which tasks are already processed
    points, _ = mm.client.scroll(
        collection_name='webarena-trajectory-history',
        limit=1000,  # Get all
        with_payload=True
    )
    
    processed_goals = {p.payload['goal'] for p in points}
    print(f"📊 Unique tasks already processed: {len(processed_goals)}")
    
    # Show a few examples
    if processed_goals:
        print("\nExample processed tasks:")
        for i, goal in enumerate(list(processed_goals)[:3]):
            print(f"  {i+1}. {goal[:80]}...")
    
    print()
    print("="*80)
    print("TO RESUME WITHOUT DUPLICATES:")
    print("="*80)
    print("Simply run the populate script again (it will auto-skip processed tasks):")
    print()
    print("python memory/populate_cues_from_trajectories.py \\")
    print("    --trajectory_dirs \\")
    print("        tmp/20251117213558_gpt_full \\")
    print("        tmp/20251118091613_gpt_full_2 \\")
    print("        tmp/20251118161129_qwen_full \\")
    print("    --collection_name webarena")
    print()
    print("The script will:")
    print(f"  • Skip {len(processed_goals)} already processed tasks")
    print("  • Process remaining tasks only")
    print("  • Handle LLM errors gracefully (skip problematic tasks)")
    print("  • Print summary at the end")
    print()
    print("To force reprocessing all tasks (creates duplicates!):")
    print("  Add --no_skip_existing flag")
    print("="*80)
    
except Exception as e:
    print(f"Error: {e}")
