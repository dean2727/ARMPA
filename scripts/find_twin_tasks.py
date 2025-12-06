"""
Find semantically similar "twin" tasks from OOD set (200-811) for each in-distribution task (0-199).

This script uses the SAME embedding model (BAAI/bge-large-en-v1.5) used in the ReasoningBank
to find semantically similar tasks.

Usage:
    python scripts/find_twin_tasks.py --output twin_tasks.json
"""

import json
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from litellm import embedding


def get_embedding(text: str, model: str = "together_ai/BAAI/bge-large-en-v1.5") -> list[float]:
    """Get embedding for a single text using litellm (same as ReasoningBank)."""
    response = embedding(model=model, input=[text])
    return response.data[0]["embedding"]


def get_embeddings_batch(texts: list[str], model: str = "together_ai/BAAI/bge-large-en-v1.5", batch_size: int = 20) -> list[list[float]]:
    """Get embeddings for multiple texts with batching and rate limiting."""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch = texts[i:i+batch_size]
        try:
            response = embedding(model=model, input=batch)
            batch_embeddings = [item["embedding"] for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error on batch {i}: {e}")
            # Retry with smaller batch or one at a time
            for text in batch:
                try:
                    time.sleep(0.5)  # Rate limit
                    emb = get_embedding(text, model)
                    all_embeddings.append(emb)
                except Exception as e2:
                    print(f"  Failed on text: {text[:50]}... Error: {e2}")
                    # Use zero vector as fallback
                    all_embeddings.append([0.0] * 1024)
        
        # Small delay between batches to avoid rate limiting
        time.sleep(0.1)
    
    return all_embeddings


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two sets of vectors."""
    # Normalize
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T


def main():
    parser = argparse.ArgumentParser(description="Find twin tasks from OOD set")
    parser.add_argument("--config_dir", type=str, default="webarena/config_files",
                        help="Path to WebArena config files")
    parser.add_argument("--output", type=str, default="twin_tasks.json",
                        help="Output file for twin task mapping")
    parser.add_argument("--embed_model", type=str, default="together_ai/BAAI/bge-large-en-v1.5",
                        help="Embedding model (should match ReasoningBank)")
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir)
    
    # Load all task intents
    print("Loading task intents...")
    in_dist_tasks = {}  # task_id -> intent
    ood_tasks = {}      # task_id -> intent
    
    for config_file in sorted(config_dir.glob("*.json")):
        try:
            task_id = int(config_file.stem)
        except ValueError:
            continue
        
        with open(config_file) as f:
            config = json.load(f)
        intent = config.get("intent", "")
        
        if 0 <= task_id <= 199:
            in_dist_tasks[task_id] = intent
        elif 200 <= task_id <= 811:
            ood_tasks[task_id] = intent
    
    print(f"  In-distribution tasks (0-199): {len(in_dist_tasks)}")
    print(f"  OOD tasks (200-811): {len(ood_tasks)}")
    
    if len(ood_tasks) == 0:
        raise ValueError("No OOD tasks found! Did you generate configs for tasks 200-811?")
    
    # Get embeddings for all tasks using the same model as ReasoningBank
    print(f"\nUsing embedding model: {args.embed_model}")
    
    print("\nComputing embeddings for in-distribution tasks...")
    in_dist_ids = sorted(in_dist_tasks.keys())
    in_dist_intents = [in_dist_tasks[tid] for tid in in_dist_ids]
    in_dist_embeddings = get_embeddings_batch(in_dist_intents, model=args.embed_model)
    
    print("\nComputing embeddings for OOD tasks...")
    ood_ids = sorted(ood_tasks.keys())
    ood_intents = [ood_tasks[tid] for tid in ood_ids]
    ood_embeddings = get_embeddings_batch(ood_intents, model=args.embed_model)
    
    # Convert to numpy arrays
    in_dist_embeddings = np.array(in_dist_embeddings)
    ood_embeddings = np.array(ood_embeddings)
    
    print(f"\nEmbedding shapes: in_dist={in_dist_embeddings.shape}, ood={ood_embeddings.shape}")
    
    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    similarities = cosine_similarity_matrix(in_dist_embeddings, ood_embeddings)
    
    # For each in-dist task, find the most similar OOD task (greedy 1-to-1 matching)
    print("Finding twin tasks (greedy 1-to-1 matching)...")
    twins = {}
    used_ood_ids = set()
    
    for i, in_dist_id in enumerate(tqdm(in_dist_ids)):
        sims = similarities[i]
        sorted_indices = np.argsort(sims)[::-1]  # Descending
        
        for j in sorted_indices:
            ood_id = ood_ids[j]
            if ood_id not in used_ood_ids:
                used_ood_ids.add(ood_id)
                twins[in_dist_id] = {
                    "ood_task_id": ood_id,
                    "similarity": float(sims[j]),
                    "in_dist_intent": in_dist_tasks[in_dist_id],
                    "ood_intent": ood_tasks[ood_id]
                }
                break
    
    # Sort twins by similarity for display
    sorted_twins = sorted(twins.items(), key=lambda x: x[1]["similarity"], reverse=True)
    
    # Print top and bottom matches
    print("\n" + "=" * 80)
    print("TOP 10 MOST SIMILAR TWINS")
    print("=" * 80)
    for in_dist_id, info in sorted_twins[:10]:
        print(f"\nIn-dist {in_dist_id}: {info['in_dist_intent'][:60]}...")
        print(f"OOD {info['ood_task_id']}: {info['ood_intent'][:60]}...")
        print(f"Similarity: {info['similarity']:.4f}")
    
    print("\n" + "=" * 80)
    print("BOTTOM 10 LEAST SIMILAR TWINS")
    print("=" * 80)
    for in_dist_id, info in sorted_twins[-10:]:
        print(f"\nIn-dist {in_dist_id}: {info['in_dist_intent'][:60]}...")
        print(f"OOD {info['ood_task_id']}: {info['ood_intent'][:60]}...")
        print(f"Similarity: {info['similarity']:.4f}")
    
    # Statistics
    similarities_list = [info["similarity"] for info in twins.values()]
    print("\n" + "=" * 80)
    print("SIMILARITY STATISTICS")
    print("=" * 80)
    print(f"Mean similarity: {np.mean(similarities_list):.4f}")
    print(f"Median similarity: {np.median(similarities_list):.4f}")
    print(f"Min similarity: {np.min(similarities_list):.4f}")
    print(f"Max similarity: {np.max(similarities_list):.4f}")
    
    # Save results
    output_data = {
        "embedding_model": args.embed_model,
        "twins": {str(k): v for k, v in twins.items()},
        "ood_task_ids": sorted([info["ood_task_id"] for info in twins.values()]),
        "statistics": {
            "mean_similarity": float(np.mean(similarities_list)),
            "median_similarity": float(np.median(similarities_list)),
            "min_similarity": float(np.min(similarities_list)),
            "max_similarity": float(np.max(similarities_list)),
            "num_twins": len(twins)
        }
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Saved twin mapping to {args.output}")
    
    # Print the OOD task IDs for easy copy-paste
    ood_ids_list = output_data["ood_task_ids"]
    print("\n" + "=" * 80)
    print(f"OOD TWIN TASK IDs ({len(ood_ids_list)} tasks)")
    print("=" * 80)
    print(",".join(map(str, ood_ids_list)))


if __name__ == "__main__":
    main()
