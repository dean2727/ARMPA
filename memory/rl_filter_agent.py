"""
RL-based Memory Filter Agent

This module implements a reinforcement learning agent that learns to select
the most useful memories for web navigation tasks. The agent uses PPO to train
a policy network that scores retrieved memories based on their predicted utility
for task completion.

Key Components:
1. MemoryFilterEnv: Custom Gym environment for memory selection
2. MemoryFilterPolicy: Neural network policy for scoring memories
3. RLMemoryFilter: Inference wrapper for trained models

Architecture:
- State: Memory embeddings + task embedding + observation embedding + entropy
- Action: Continuous scores [0, 1] for each memory
- Reward: Task success (1.0) or failure (0.0)

Author: ARMPA Team
Date: November 2025
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle


class MemoryFilterPolicy(nn.Module):
    """
    Neural network policy for scoring memories.
    
    Architecture:
        - Shared encoder processes concatenated state features
        - Per-memory scoring head outputs relevance scores [0, 1]
    
    Input:
        - Memory embeddings: (num_memories, embedding_dim)
        - Task embedding: (embedding_dim,)
        - Observation embedding: (embedding_dim,)
        - Entropy: (1,)
    
    Output:
        - Memory scores: (num_memories,) values in [0, 1]
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        max_memories: int = 10,
        hidden_dim: int = 256,
    ):
        """
        Initialize the memory filter policy network.
        
        Args:
            embedding_dim: Dimension of embeddings (from MemoryManager)
            max_memories: Maximum number of memories to score
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.hidden_dim = hidden_dim
        
        # Calculate total input dimension
        # Context: task_emb + obs_emb + entropy
        context_dim = embedding_dim * 2 + 1
        # Per-memory: memory_emb
        per_memory_dim = embedding_dim
        
        # Shared context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Per-memory encoder (processes each memory embedding)
        self.memory_encoder = nn.Sequential(
            nn.Linear(per_memory_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Scoring head (combines context + memory features)
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(
        self,
        memory_embeddings: torch.Tensor,  # (batch, num_memories, emb_dim)
        task_embedding: torch.Tensor,      # (batch, emb_dim)
        obs_embedding: torch.Tensor,       # (batch, emb_dim)
        entropy: torch.Tensor,             # (batch, 1)
        num_memories: torch.Tensor,        # (batch,) actual number of memories
    ) -> torch.Tensor:
        """
        Forward pass to score memories.
        
        Args:
            memory_embeddings: Embeddings of retrieved memories
            task_embedding: Embedding of task/goal
            obs_embedding: Embedding of current observation
            entropy: Agent's current entropy
            num_memories: Actual number of memories (for masking padding)
        
        Returns:
            scores: (batch, num_memories) relevance scores in [0, 1]
        """
        batch_size = memory_embeddings.shape[0]
        num_mem = memory_embeddings.shape[1]
        
        # Encode context (task + observation + entropy)
        context = torch.cat([task_embedding, obs_embedding, entropy], dim=-1)
        context_features = self.context_encoder(context)  # (batch, hidden_dim)
        
        # Expand context for broadcasting with each memory
        context_features = context_features.unsqueeze(1).expand(
            batch_size, num_mem, self.hidden_dim
        )  # (batch, num_memories, hidden_dim)
        
        # Encode each memory
        memory_features = self.memory_encoder(
            memory_embeddings
        )  # (batch, num_memories, hidden_dim)
        
        # Combine context + memory features and score
        combined = torch.cat(
            [context_features, memory_features], dim=-1
        )  # (batch, num_memories, hidden_dim * 2)
        
        scores = self.scoring_head(combined).squeeze(-1)  # (batch, num_memories)
        
        # Mask out padding (memories beyond actual count)
        mask = torch.arange(num_mem, device=scores.device).expand(
            batch_size, num_mem
        ) < num_memories.unsqueeze(1)
        scores = scores * mask.float()
        
        return scores


class MemoryFilterEnv(gym.Env):
    """
    Custom Gymnasium environment for training memory filter agent.
    
    This environment wraps a single WebArena task episode. At each step,
    the agent receives retrieved memories and must score them. The selected
    memories are then provided to the main WebArena agent. The reward is
    the final task success.
    
    Episode flow:
    1. Reset: Initialize WebArena task
    2. Step: Receive memories, output scores, execute main agent with filtered memories
    3. Repeat until task completion
    4. Return reward based on success
    """
    
    def __init__(
        self,
        config_files: List[str],
        memory_manager,
        webarena_agent,
        max_steps: int = 30,
        embedding_dim: int = 384,
        max_memories: int = 10,
    ):
        """
        Initialize the memory filter environment.
        
        Args:
            config_files: List of WebArena task config files
            memory_manager: MemoryManager instance for retrieval
            webarena_agent: WebArena agent for task execution
            max_steps: Maximum steps per episode
            embedding_dim: Embedding dimension
            max_memories: Maximum number of memories
        """
        super().__init__()
        
        self.config_files = config_files
        self.memory_manager = memory_manager
        self.webarena_agent = webarena_agent
        self.max_steps = max_steps
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        
        # Define observation space (state)
        # Components: memory_embs + task_emb + obs_emb + entropy + num_memories
        self.observation_space = gym.spaces.Dict({
            'memory_embeddings': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_memories, embedding_dim),
                dtype=np.float32
            ),
            'task_embedding': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(embedding_dim,),
                dtype=np.float32
            ),
            'obs_embedding': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(embedding_dim,),
                dtype=np.float32
            ),
            'entropy': gym.spaces.Box(
                low=0, high=10,
                shape=(1,),
                dtype=np.float32
            ),
            'num_memories': gym.spaces.Discrete(max_memories + 1),
        })
        
        # Define action space (memory scores)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(max_memories,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_task_idx = 0
        self.step_count = 0
        self.current_task = None
        self.task_success = False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset environment for a new episode.
        
        Returns:
            observation: Initial state
            info: Additional info
        """
        super().reset(seed=seed)
        
        # Select next task (cycle through config files)
        self.current_task_idx = (self.current_task_idx + 1) % len(self.config_files)
        self.current_task = self.config_files[self.current_task_idx]
        self.step_count = 0
        self.task_success = False
        
        # Initialize WebArena episode
        # TODO: Actually initialize WebArena env - placeholder for now
        
        # Return initial observation (placeholder - will be set on first step)
        obs = {
            'memory_embeddings': np.zeros((self.max_memories, self.embedding_dim), dtype=np.float32),
            'task_embedding': np.zeros(self.embedding_dim, dtype=np.float32),
            'obs_embedding': np.zeros(self.embedding_dim, dtype=np.float32),
            'entropy': np.array([0.0], dtype=np.float32),
            'num_memories': 0,
        }
        
        info = {'task': self.current_task}
        
        return obs, info
    
    def step(self, action: np.ndarray):
        """
        Execute one step: filter memories and run WebArena agent.
        
        Args:
            action: Memory scores from RL policy
        
        Returns:
            observation: Next state
            reward: Task success reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional info
        """
        self.step_count += 1
        
        # TODO: This is a placeholder. Real implementation will:
        # 1. Apply action (memory scores) to filter memories
        # 2. Execute WebArena agent with filtered memories
        # 3. Get next observation from WebArena
        # 4. Check if task completed
        # 5. Return appropriate reward
        
        # Placeholder: random success, sparse reward at end
        terminated = self.step_count >= self.max_steps or np.random.random() < 0.1
        self.task_success = np.random.random() < 0.3  # 30% random success for testing
        
        reward = 1.0 if (terminated and self.task_success) else 0.0
        truncated = self.step_count >= self.max_steps
        
        # Next observation (placeholder)
        obs = {
            'memory_embeddings': np.random.randn(self.max_memories, self.embedding_dim).astype(np.float32),
            'task_embedding': np.random.randn(self.embedding_dim).astype(np.float32),
            'obs_embedding': np.random.randn(self.embedding_dim).astype(np.float32),
            'entropy': np.array([np.random.random()], dtype=np.float32),
            'num_memories': np.random.randint(1, self.max_memories + 1),
        }
        
        info = {
            'step': self.step_count,
            'task_success': self.task_success,
        }
        
        return obs, reward, terminated, truncated, info


class RLMemoryFilter:
    """
    Inference wrapper for trained RL memory filter agent.
    
    This class loads a trained policy and provides a simple interface
    for filtering memories in production (integrated into run.py).
    """
    
    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.6,
        device: str = "mps",
        embedding_dim: int = 384,
        max_memories: int = 10,
    ):
        """
        Initialize the RL memory filter.
        
        Args:
            model_path: Path to trained model checkpoint
            score_threshold: Minimum score to include memory
            device: Device for inference (mps, cuda, cpu)
            embedding_dim: Embedding dimension
            max_memories: Maximum memories to score
        """
        self.score_threshold = score_threshold
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        
        # Load trained policy
        self.policy = MemoryFilterPolicy(
            embedding_dim=embedding_dim,
            max_memories=max_memories,
        ).to(self.device)
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy.eval()
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    def filter_memories(
        self,
        memories: List[Dict[str, Any]],
        task_embedding: np.ndarray,
        obs_embedding: np.ndarray,
        entropy: float,
    ) -> List[Dict[str, Any]]:
        """
        Filter memories using trained RL policy.
        
        Args:
            memories: List of retrieved memories from MemoryManager
            task_embedding: Embedding of task/goal
            obs_embedding: Embedding of current observation
            entropy: Current agent entropy
        
        Returns:
            filtered_memories: Memories with score >= threshold, sorted by score
        """
        if len(memories) == 0:
            return []
        
        # Extract memory embeddings
        memory_embeddings = np.array([m['embedding'] for m in memories])
        num_memories = len(memories)
        
        # Pad to max_memories if needed
        if num_memories < self.max_memories:
            padding = np.zeros(
                (self.max_memories - num_memories, self.embedding_dim),
                dtype=np.float32
            )
            memory_embeddings = np.vstack([memory_embeddings, padding])
        
        # Convert to tensors
        memory_emb_tensor = torch.from_numpy(memory_embeddings).unsqueeze(0).to(self.device)
        task_emb_tensor = torch.from_numpy(task_embedding).unsqueeze(0).to(self.device)
        obs_emb_tensor = torch.from_numpy(obs_embedding).unsqueeze(0).to(self.device)
        entropy_tensor = torch.tensor([[entropy]], dtype=torch.float32).to(self.device)
        num_mem_tensor = torch.tensor([num_memories], dtype=torch.long).to(self.device)
        
        # Get scores from policy
        with torch.no_grad():
            scores = self.policy(
                memory_emb_tensor,
                task_emb_tensor,
                obs_emb_tensor,
                entropy_tensor,
                num_mem_tensor,
            )[0].cpu().numpy()
        
        # Filter and sort memories by score
        scored_memories = [
            (mem, score) for mem, score in zip(memories, scores[:num_memories])
            if score >= self.score_threshold
        ]
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        filtered_memories = [mem for mem, score in scored_memories]
        
        return filtered_memories
    
    def get_stats(
        self,
        memories: List[Dict[str, Any]],
        filtered_memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get filtering statistics for logging.
        
        Args:
            memories: Original memories
            filtered_memories: Filtered memories
        
        Returns:
            stats: Dictionary with filtering statistics
        """
        return {
            'original_count': len(memories),
            'filtered_count': len(filtered_memories),
            'threshold': self.score_threshold,
            'filter_ratio': len(filtered_memories) / len(memories) if memories else 0.0,
        }


# Testing utilities
if __name__ == "__main__":
    print("Testing MemoryFilterPolicy...")
    
    # Test policy network
    policy = MemoryFilterPolicy(embedding_dim=384, max_memories=10, hidden_dim=256)
    
    # Create dummy inputs
    batch_size = 4
    num_memories = 5
    memory_embs = torch.randn(batch_size, num_memories, 384)
    task_emb = torch.randn(batch_size, 384)
    obs_emb = torch.randn(batch_size, 384)
    entropy = torch.rand(batch_size, 1)
    num_mem = torch.tensor([5, 3, 4, 5])
    
    # Forward pass
    scores = policy(memory_embs, task_emb, obs_emb, entropy, num_mem)
    
    print(f"Input shape: {memory_embs.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Scores range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"Sample scores: {scores[0]}")
    print("\n✓ MemoryFilterPolicy test passed!")
    
    print("\nTesting MemoryFilterEnv...")
    # Note: Full env testing requires WebArena setup
    print("✓ Environment structure defined (requires WebArena for full test)")
