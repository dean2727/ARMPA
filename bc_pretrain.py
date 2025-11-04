"""
Behavioral Cloning Pretraining for RL Memory Filter

Trains a supervised model to predict which memories are useful based on
successful episodes. This provides a warm start for RL fine-tuning.

Usage:
    python bc_pretrain.py --data_dir runs/webshop/results_*/rl_training_data \
                          --output_dir models/bc_pretrained \
                          --epochs 20
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

from memory.rl_filter_agent import MemoryFilterPolicy


class MemorySelectionDataset(Dataset):
    """Dataset for behavioral cloning on successful episodes"""
    
    def __init__(self, episode_files, use_successful_only=True):
        self.samples = []
        
        print(f"Loading episodes from {len(episode_files)} files...")
        for ep_file in tqdm(episode_files):
            with open(ep_file, 'rb') as f:
                episode = pickle.load(f)
            
            # Episodes are lists of steps; check if last step has reward=1.0
            if not isinstance(episode, list) or len(episode) == 0:
                continue
                
            # Only use successful episodes for BC
            success = episode[-1].get('reward', 0.0) == 1.0
            if use_successful_only and not success:
                continue
            
            # Extract step data - episode is already the list of steps
            for step_data in episode:
                if step_data['memory_embeddings'] is None or len(step_data['memory_embeddings']) == 0:
                    continue
                
                # Create training sample
                sample = {
                    'memory_embeddings': torch.FloatTensor(step_data['memory_embeddings']),
                    'task_embedding': torch.FloatTensor(step_data['task_embedding']),
                    'obs_embedding': torch.FloatTensor(step_data['obs_embedding']),
                    'entropy': step_data['entropy'],
                    'reward': step_data.get('reward', 0.0),
                    # Target: all memories are "useful" in successful episodes
                    'target_scores': torch.ones(len(step_data['memory_embeddings'])),
                }
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} training samples from successful episodes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate to handle variable number of memories"""
    # Pad memories to max length in batch
    max_memories = max(s['memory_embeddings'].shape[0] for s in batch)
    
    batch_data = {
        'memory_embeddings': [],
        'task_embedding': [],
        'obs_embedding': [],
        'entropy': [],
        'reward': [],
        'target_scores': [],
        'mask': [],
        'num_memories': [],
    }
    
    for sample in batch:
        num_mems = sample['memory_embeddings'].shape[0]
        pad_size = max_memories - num_mems
        
        # Pad memory embeddings and targets
        if pad_size > 0:
            mem_pad = torch.zeros(pad_size, sample['memory_embeddings'].shape[1])
            target_pad = torch.zeros(pad_size)
            
            batch_data['memory_embeddings'].append(torch.cat([sample['memory_embeddings'], mem_pad], dim=0))
            batch_data['target_scores'].append(torch.cat([sample['target_scores'], target_pad], dim=0))
            
            # Create mask (1 for real memories, 0 for padding)
            mask = torch.cat([torch.ones(num_mems), torch.zeros(pad_size)])
        else:
            batch_data['memory_embeddings'].append(sample['memory_embeddings'])
            batch_data['target_scores'].append(sample['target_scores'])
            mask = torch.ones(num_mems)
        
        batch_data['mask'].append(mask)
        batch_data['task_embedding'].append(sample['task_embedding'])
        batch_data['obs_embedding'].append(sample['obs_embedding'])
        batch_data['entropy'].append(sample['entropy'])
        batch_data['reward'].append(sample['reward'])
        batch_data['num_memories'].append(num_mems)
    
    return {
        'memory_embeddings': torch.stack(batch_data['memory_embeddings']),
        'task_embedding': torch.stack(batch_data['task_embedding']),
        'obs_embedding': torch.stack(batch_data['obs_embedding']),
        'entropy': torch.FloatTensor(batch_data['entropy']).unsqueeze(-1),  # (batch, 1)
        'reward': torch.FloatTensor(batch_data['reward']),
        'target_scores': torch.stack(batch_data['target_scores']),
        'mask': torch.stack(batch_data['mask']),
        'num_memories': torch.LongTensor(batch_data['num_memories']),
    }


def train_bc(model, train_loader, val_loader, num_epochs, lr, device, save_dir):
    """Train model with behavioral cloning"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')  # Per-element loss
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            pred_scores = model(
                batch['memory_embeddings'],
                batch['task_embedding'],
                batch['obs_embedding'],
                batch['entropy'],
                batch['num_memories']
            )
            
            # Compute loss (only on non-padded memories)
            loss = criterion(pred_scores, batch['target_scores'])
            loss = (loss * batch['mask']).sum() / batch['mask'].sum()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                pred_scores = model(
                    batch['memory_embeddings'],
                    batch['task_embedding'],
                    batch['obs_embedding'],
                    batch['entropy'],
                    batch['num_memories']
                )
                
                loss = criterion(pred_scores, batch['target_scores'])
                loss = (loss * batch['mask']).sum() / batch['mask'].sum()
                val_loss += loss.item()
        
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
            val_loss_str = f"{val_loss:.4f}"
        else:
            val_loss_str = "N/A"
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss_str}")
        
        # Save best model (use train loss if no validation set)
        metric_to_compare = val_loss if len(val_loader) > 0 else train_loss
        if metric_to_compare < best_val_loss:
            best_val_loss = val_loss
            save_path = save_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Behavioral Cloning Pretraining")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing rl_training_data subdirectories')
    parser.add_argument('--output_dir', type=str, default='models/bc_pretrained',
                        help='Directory to save trained model')
    parser.add_argument('--embed_dim', type=int, default=384,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation set fraction')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device (cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use CPU for M1 Mac compatibility (MPS not available in this PyTorch version)
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Find all episode files
    data_dir = Path(args.data_dir)
    episode_files = []
    
    if data_dir.is_dir() and (data_dir / 'rl_training_data').exists():
        # Single results directory
        episode_files = list((data_dir / 'rl_training_data').glob('episode_*.pkl'))
    else:
        # Multiple results directories (pattern matching)
        for results_dir in Path('runs/webshop').glob('results_*/rl_training_data'):
            episode_files.extend(results_dir.glob('episode_*.pkl'))
    
    if not episode_files:
        print(f"❌ No episode files found in {data_dir}")
        return
    
    print(f"Found {len(episode_files)} episode files")
    
    # Create dataset
    dataset = MemorySelectionDataset(episode_files, use_successful_only=True)
    
    if len(dataset) == 0:
        print("❌ No valid training samples found!")
        return
    
    # Split train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Create model
    model = MemoryFilterPolicy(
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    model = train_bc(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=output_dir,
    )
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Training complete! Models saved to {output_dir}")
    
    # Save training config
    config = vars(args)
    config['num_episodes'] = len(episode_files)
    config['num_samples'] = len(dataset)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
