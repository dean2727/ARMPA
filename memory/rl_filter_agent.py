"""
RL Memory Filter Agent - Per-Memory Gating with GRPO

This module implements the episodic RL memory filter agent as specified in
"ARMPA Memory-Filter Agent Formulation.md". The agent learns to gate individual
memories using sigmoid outputs, trained with Group Relative Policy Optimization (GRPO).

Key Features:
- Per-memory sigmoid gating (not binary selection)
- Episodic reward assignment (success + efficiency bonus)
- GRPO policy updates with group-relative advantage
- Modular design for easy integration and testing

Author: ARMPA Research Team
Date: 2025-11-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class MemoryFilterNetwork(nn.Module):
    """
    MLP that outputs sigmoid gate probability for each (context, memory) pair.
    
    Architecture:
        Input: [task_emb (1024) | obs_emb (1024) | memory_emb (1024) | entropy (1)] = 3073-dim
        Hidden: 2 MLP layers with ReLU
        Output: 1 sigmoid probability (gate score)
    """
    
    def __init__(
        self,
        task_dim: int = 1024,
        obs_dim: int = 1024,
        memory_dim: int = 1024,
        entropy_dim: int = 1,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.task_dim = task_dim
        self.obs_dim = obs_dim
        self.memory_dim = memory_dim
        self.entropy_dim = entropy_dim
        
        # Input dimension
        input_dim = task_dim + obs_dim + memory_dim + entropy_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer (single sigmoid gate)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        task_emb: torch.Tensor,      # (batch, task_dim)
        obs_emb: torch.Tensor,       # (batch, obs_dim)
        memory_emb: torch.Tensor,    # (batch, memory_dim)
        entropy: torch.Tensor,       # (batch, 1)
    ) -> torch.Tensor:
        """
        Compute gate probability for each (context, memory) pair.
        
        Args:
            task_emb: Task/goal embedding
            obs_emb: Current observation embedding
            memory_emb: Candidate memory embedding
            entropy: Uncertainty measure (e.g., mean entropy)
        
        Returns:
            gate_prob: Sigmoid probability in [0, 1], shape (batch, 1)
        """
        # Concatenate all inputs
        x = torch.cat([task_emb, obs_emb, memory_emb, entropy], dim=-1)
        
        # Forward through network
        logits = self.network(x)
        
        # Apply sigmoid to get probability
        gate_prob = torch.sigmoid(logits)
        
        return gate_prob
    
    def compute_log_prob(
        self,
        task_emb: torch.Tensor,
        obs_emb: torch.Tensor,
        memory_emb: torch.Tensor,
        entropy: torch.Tensor,
        action: torch.Tensor,  # Binary gate decision (0 or 1)
    ) -> torch.Tensor:
        """
        Compute log probability of a binary gate decision.
        
        This is used for policy gradient computation. The action is treated
        as a sample from a Bernoulli distribution with p = gate_prob.
        
        Args:
            task_emb, obs_emb, memory_emb, entropy: Context features
            action: Binary decision (0 = reject, 1 = accept), shape (batch, 1)
        
        Returns:
            log_prob: Log probability of the action, shape (batch, 1)
        """
        gate_prob = self.forward(task_emb, obs_emb, memory_emb, entropy)
        
        # Clamp probabilities to avoid log(0)
        gate_prob = torch.clamp(gate_prob, min=1e-8, max=1-1e-8)
        
        # Bernoulli log probability
        log_prob = action * torch.log(gate_prob) + (1 - action) * torch.log(1 - gate_prob)
        
        return log_prob


class RLMemoryFilter:
    """
    RL-based memory filter using per-memory gating with GRPO.
    
    Training Procedure:
    1. Collect N episodes (trajectories) with memory retrieval
    2. For each episode, log all recall events with (context, candidates, gates)
    3. At episode end, assign episodic reward: r = success + γ(1 - steps/max_steps)
    4. Compute group-relative advantage across N episodes
    5. Update policy with PPO-style clipped objective
    
    Inference Procedure:
    1. Given K candidate memories from similarity search
    2. For each memory, compute gate probability g_i = π(1 | context, memory_i)
    3. Select all memories with g_i > threshold (e.g., 0.5)
    4. Return filtered subset (can be empty, partial, or all)
    """
    
    def __init__(
        self,
        task_dim: int = 1024,
        obs_dim: int = 1024,
        memory_dim: int = 1024,
        hidden_dims: List[int] = [512, 256],
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.01,
        gamma: float = 0.5,  # Reward shaping for step efficiency
        max_steps: int = 30,
        score_threshold: float = 0.5,
        device: str = "cpu",
        model_path: Optional[str] = None,
    ):
        """
        Initialize RL memory filter.
        
        Args:
            task_dim: Dimension of task embedding
            obs_dim: Dimension of observation embedding
            memory_dim: Dimension of memory embedding
            hidden_dims: Hidden layer dimensions for MLP
            learning_rate: Learning rate for optimizer
            clip_epsilon: PPO clipping parameter
            kl_beta: KL divergence penalty coefficient
            gamma: Weight for step efficiency in reward
            max_steps: Maximum steps per episode (for reward normalization)
            score_threshold: Gate probability threshold for memory selection
            device: Device for computation ("cpu", "cuda", "mps")
            model_path: Path to pre-trained model (if None, initialize randomly)
        """
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.gamma = gamma
        self.max_steps = max_steps
        self.clip_epsilon = clip_epsilon
        self.kl_beta = kl_beta
        
        # Initialize network
        self.policy_net = MemoryFilterNetwork(
            task_dim=task_dim,
            obs_dim=obs_dim,
            memory_dim=memory_dim,
            entropy_dim=1,
            hidden_dims=hidden_dims,
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Load pre-trained model if provided
        if model_path is not None:
            self.load_model(model_path)
            logger.info(f"[RL Filter] Loaded model from {model_path}")
        
        # Episode buffer for training
        self.episode_buffer = []
    
    def filter_memories(
        self,
        memories: List[Dict[str, Any]],
        task_embedding: np.ndarray,
        obs_embedding: np.ndarray,
        entropy: float,
        return_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Filter candidate memories using learned gating policy (inference mode).
        
        Args:
            memories: List of K candidate memories from similarity search
            task_embedding: Task/goal embedding (1024-dim)
            obs_embedding: Current observation embedding (1024-dim)
            entropy: Uncertainty measure (scalar)
            return_scores: If True, add 'gate_score' to each memory dict
        
        Returns:
            filtered_memories: Subset of memories where gate > threshold
        """
        if len(memories) == 0:
            return []
        
        # Convert to tensors
        task_emb = torch.tensor(task_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_emb = torch.tensor(obs_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        entropy_tensor = torch.tensor([[entropy]], dtype=torch.float32).to(self.device)
        
        # Compute gate scores for all candidates
        filtered = []
        self.policy_net.eval()
        with torch.no_grad():
            for memory in memories:
                # Get memory embedding
                mem_emb_np = memory.get('embedding')
                if mem_emb_np is None:
                    logger.warning(f"[RL Filter] Memory {memory.get('memory_id')} missing embedding, skipping")
                    continue
                
                mem_emb = torch.tensor(mem_emb_np, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Compute gate probability
                gate_prob = self.policy_net(task_emb, obs_emb, mem_emb, entropy_tensor)
                gate_score = gate_prob.item()
                
                # Add score to memory dict if requested
                if return_scores:
                    memory['gate_score'] = gate_score
                
                # Select if above threshold
                if gate_score > self.score_threshold:
                    filtered.append(memory)
        
        return filtered
    
    def log_recall_event(
        self,
        task_embedding: np.ndarray,
        obs_embedding: np.ndarray,
        entropy: float,
        candidates: List[Dict[str, Any]],
        gate_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Log a recall event for training data collection.
        
        This is called during episode execution when memory retrieval happens.
        Stores context and candidate information for later policy update.
        
        Args:
            task_embedding: Task embedding (1024-dim)
            obs_embedding: Observation embedding (1024-dim)
            entropy: Uncertainty measure
            candidates: List of K candidate memories
            gate_scores: Optional pre-computed gate scores (for inference logging)
        
        Returns:
            recall_data: Dictionary containing all recall event information
        """
        recall_data = {
            'task_embedding': task_embedding,  # np.ndarray
            'obs_embedding': obs_embedding,    # np.ndarray
            'entropy': entropy,                # float
            'candidates': [],
        }
        
        for i, candidate in enumerate(candidates):
            candidate_data = {
                'memory_id': candidate.get('memory_id'),
                'embedding': candidate.get('embedding'),  # np.ndarray
                'similarity_score': candidate.get('score'),
                'gate_score': gate_scores[i] if gate_scores else None,
                'selected': None,  # Filled during inference or training
            }
            recall_data['candidates'].append(candidate_data)
        
        return recall_data
    
    def add_episode(
        self,
        recall_events: List[Dict[str, Any]],
        final_reward: float,
        success: bool,
        num_steps: int,
    ):
        """
        Add a complete episode to the buffer for training.
        
        Args:
            recall_events: List of recall events from the episode
            final_reward: Episodic reward (success + efficiency bonus)
            success: Whether episode succeeded
            num_steps: Number of steps taken
        """
        episode_data = {
            'recall_events': recall_events,
            'final_reward': final_reward,
            'success': success,
            'num_steps': num_steps,
        }
        self.episode_buffer.append(episode_data)
    
    def compute_episodic_reward(self, success: bool, num_steps: int) -> float:
        """
        Compute episodic reward with step efficiency bonus.
        
        Formula: r = 1_{success} + γ * (1 - steps/max_steps) * 1_{success}
        
        Args:
            success: Whether episode succeeded (1 or 0)
            num_steps: Number of steps taken
        
        Returns:
            reward: Episodic reward (scalar)
        """
        if success:
            efficiency_bonus = self.gamma * (1.0 - min(num_steps, self.max_steps) / self.max_steps)
            reward = 1.0 + efficiency_bonus
        else:
            reward = 0.0
        
        return reward
    
    def update_policy_grpo(
        self,
        episodes: Optional[List[Dict[str, Any]]] = None,
        normalize_advantages: bool = True,
    ) -> Dict[str, float]:
        """
        Update policy using Group Relative Policy Optimization (GRPO).
        
        GRPO Procedure:
        1. For each trajectory n, compute advantage: A^n = (r^n - r_mean) / (r_std + δ)
        2. Aggregate objective across all (T recall events × K candidates)
        3. Use PPO-style clipped objective with KL penalty
        4. Update policy parameters
        
        Args:
            episodes: List of episode dicts (if None, use self.episode_buffer)
            normalize_advantages: If True, normalize advantages with mean/std
        
        Returns:
            metrics: Dictionary of training metrics (loss, KL, etc.)
        """
        if episodes is None:
            episodes = self.episode_buffer
        
        if len(episodes) == 0:
            logger.warning("[RL Filter] No episodes in buffer, skipping update")
            return {}
        
        # Extract rewards and compute group-relative advantages
        rewards = np.array([ep['final_reward'] for ep in episodes])
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        
        advantages = (rewards - mean_reward) / std_reward
        
        # Optionally normalize advantages again (helps with stability)
        if normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Collect all (context, memory, advantage) tuples across episodes
        batch_data = []
        for ep_idx, episode in enumerate(episodes):
            advantage = advantages[ep_idx]
            
            for recall_event in episode['recall_events']:
                task_emb = recall_event['task_embedding']
                obs_emb = recall_event['obs_embedding']
                entropy = recall_event['entropy']
                
                for candidate in recall_event['candidates']:
                    mem_emb = candidate['embedding']
                    
                    # For GRPO, we need to sample gate decisions
                    # During training, we treat ALL gates as having been "executed"
                    # Gate decision is binary: 1 = accept, 0 = reject
                    # We'll sample from current policy for each candidate
                    batch_data.append({
                        'task_emb': task_emb,
                        'obs_emb': obs_emb,
                        'memory_emb': mem_emb,
                        'entropy': entropy,
                        'advantage': advantage,
                    })
        
        if len(batch_data) == 0:
            logger.warning("[RL Filter] No candidate memories in episodes, skipping update")
            return {}
        
        # Convert to tensors
        task_embs = torch.tensor(
            np.stack([d['task_emb'] for d in batch_data]), dtype=torch.float32
        ).to(self.device)
        obs_embs = torch.tensor(
            np.stack([d['obs_emb'] for d in batch_data]), dtype=torch.float32
        ).to(self.device)
        memory_embs = torch.tensor(
            np.stack([d['memory_emb'] for d in batch_data]), dtype=torch.float32
        ).to(self.device)
        entropies = torch.tensor(
            [[d['entropy']] for d in batch_data], dtype=torch.float32
        ).to(self.device)
        advantages_tensor = torch.tensor(
            [[d['advantage']] for d in batch_data], dtype=torch.float32
        ).to(self.device)
        
        # Compute old policy probabilities (for KL divergence)
        self.policy_net.eval()
        with torch.no_grad():
            old_gate_probs = self.policy_net(task_embs, obs_embs, memory_embs, entropies)
            old_gate_probs = torch.clamp(old_gate_probs, min=1e-8, max=1-1e-8)
        
        # Sample gate actions from old policy (for GRPO)
        # In GRPO, we sample actions and use them for gradient estimation
        gate_actions = (torch.rand_like(old_gate_probs) < old_gate_probs).float()
        
        # Policy update
        self.policy_net.train()
        
        # Forward pass with current policy
        gate_probs = self.policy_net(task_embs, obs_embs, memory_embs, entropies)
        gate_probs = torch.clamp(gate_probs, min=1e-8, max=1-1e-8)
        
        # Compute log probabilities
        log_probs = gate_actions * torch.log(gate_probs) + (1 - gate_actions) * torch.log(1 - gate_probs)
        old_log_probs = gate_actions * torch.log(old_gate_probs) + (1 - gate_actions) * torch.log(1 - old_gate_probs)
        
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # PPO clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages_tensor, clipped_ratio * advantages_tensor).mean()
        
        # KL divergence penalty (between old and new policies)
        kl_div = (old_gate_probs * torch.log(old_gate_probs / gate_probs) + 
                  (1 - old_gate_probs) * torch.log((1 - old_gate_probs) / (1 - gate_probs))).mean()
        
        # Total loss
        loss = policy_loss + self.kl_beta * kl_div
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_advantage': advantages.mean(),
            'mean_gate_prob': gate_probs.mean().item(),
            'num_episodes': len(episodes),
            'num_samples': len(batch_data),
        }
        
        # Clear episode buffer after update
        self.episode_buffer = []
        
        return metrics
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'score_threshold': self.score_threshold,
                'gamma': self.gamma,
                'max_steps': self.max_steps,
                'clip_epsilon': self.clip_epsilon,
                'kl_beta': self.kl_beta,
            }
        }, path)
        logger.info(f"[RL Filter] Saved model to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.score_threshold = config.get('score_threshold', self.score_threshold)
            self.gamma = config.get('gamma', self.gamma)
            self.max_steps = config.get('max_steps', self.max_steps)
            self.clip_epsilon = config.get('clip_epsilon', self.clip_epsilon)
            self.kl_beta = config.get('kl_beta', self.kl_beta)
        
        logger.info(f"[RL Filter] Loaded model from {path}")


def test_rl_filter():
    """Simple test of RLMemoryFilter functionality."""
    print("Testing RL Memory Filter...")
    
    # Initialize filter
    filter_agent = RLMemoryFilter(
        task_dim=1024,
        obs_dim=1024,
        memory_dim=1024,
        device="cpu",
    )
    
    # Simulate candidate memories
    np.random.seed(42)
    memories = [
        {
            'memory_id': f'mem_{i}',
            'embedding': np.random.randn(1024),
            'score': 0.9 - i * 0.1,
        }
        for i in range(5)
    ]
    
    # Simulate context
    task_emb = np.random.randn(1024)
    obs_emb = np.random.randn(1024)
    entropy = 0.5
    
    # Test filtering
    filtered = filter_agent.filter_memories(
        memories=memories,
        task_embedding=task_emb,
        obs_embedding=obs_emb,
        entropy=entropy,
        return_scores=True,
    )
    
    print(f"Original: {len(memories)} memories")
    print(f"Filtered: {len(filtered)} memories")
    for mem in filtered:
        print(f"  - {mem['memory_id']}: gate={mem['gate_score']:.3f}, sim={mem['score']:.3f}")
    
    # Test training
    print("\nTesting policy update...")
    
    # Create dummy episodes
    for ep in range(3):
        recall_events = []
        for step in range(2):
            recall_data = filter_agent.log_recall_event(
                task_embedding=np.random.randn(1024),
                obs_embedding=np.random.randn(1024),
                entropy=0.5,
                candidates=memories[:3],
            )
            recall_events.append(recall_data)
        
        filter_agent.add_episode(
            recall_events=recall_events,
            final_reward=filter_agent.compute_episodic_reward(success=(ep % 2 == 0), num_steps=10),
            success=(ep % 2 == 0),
            num_steps=10,
        )
    
    # Update policy
    metrics = filter_agent.update_policy_grpo()
    print("Update metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test save/load
    print("\nTesting save/load...")
    filter_agent.save_model("/tmp/test_rl_filter.pt")
    
    filter_agent_loaded = RLMemoryFilter(device="cpu", model_path="/tmp/test_rl_filter.pt")
    print("Model loaded successfully!")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_rl_filter()
